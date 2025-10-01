import sys
import json
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.hubert_ecg import HuBERTECGConfig, HuBERTECG as HuBERT
from HuBert_ECG.hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification
from HuBert_ECG.dataset import ECGDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cardio_learning_labels = pd.read_pickle('./reproducibility/cardio-learning-labels.pkl')
    num_labels = cardio_learning_labels[3:].shape[0]

    args = {
        'model_path': "./output/model_weights/hubert_ecg_large_weights_only.pt",
        'path_to_dataset_csv': './reproducibility/ptbxl/ptbxl_all_test.csv',
        'ecg_dir_path': './output/PTB-XL',
        'num_labels': num_labels,
        'label_start_index': 4,
        'tta': False,
        'tta_aggregation': 'mean',
        'downsampling_factor': 5,
        'batch_size': 64,
        'task': 'multi_label',
        # 'task': 'multi_class',
        'return_full_length': True,
    }
    print(num_labels) # 164
    
    # Load the JSON
    with open("./configs/model_config.json", 'r') as f:
        config_dict = json.load(f)

    config = HuBERTECGConfig(**config_dict)
    config.conv_pos_batch_norm = False
    use_label_embedding = False
    
    print({
        'num_labels': args['num_labels'],
        'classifier_hidden_size': config.classifier_proj_size,
        'use_label_embedding': use_label_embedding,
    })
    # {'num_labels': 164, 'classifier_hidden_size': 512, 'use_label_embedding': False}
    
    pretrained_hubert = HuBERT(config)

    hubert = HuBERTClassification(
        pretrained_hubert,
        num_labels=args['num_labels'],
        classifier_hidden_size=config.classifier_proj_size,
        use_label_embedding=use_label_embedding
    ).to(device)
    hubert.eval()

    new_model_state_dict = {
        k:v for k, v in hubert.state_dict().items() 
        if k in ["classifier.0.weight", "classifier.0.bias",  "classifier.2.weight",  "classifier.2.bias"]}
    print(new_model_state_dict.keys())

    checkpoint = torch.load(args['model_path'], map_location='cpu', weights_only=True)
    model_state_dict = checkpoint['model_state_dict']

    for k, v in model_state_dict.items():
        if k in ["final_proj.0.weight", "final_proj.0.bias", "label_embedding.0.weight"]:
            continue
        new_key = k
        # with this changes - load_state_dict fails 
        # if k.endswith("parametrizations.weight.original0"):
        #     new_key = k.replace("parametrizations.weight.original0", "weight_g")
        # elif k.endswith("parametrizations.weight.original1"):
        #     new_key = k.replace("parametrizations.weight.original1", "weight_v")

        new_model_state_dict[f"hubert_ecg.{new_key}"] = v

    hubert.load_state_dict(new_model_state_dict)


    testset = ECGDataset(
        path_to_dataset_csv = args['path_to_dataset_csv'],
        ecg_dir_path = args['ecg_dir_path'],
        pretrain = False,
        downsampling_factor=args['downsampling_factor'],
        label_start_index=args['label_start_index'],
        return_full_length=args['return_full_length'],
    )

    dataloader = DataLoader(
        testset,
        collate_fn=testset.collate,
        num_workers=5,
        batch_size = args['batch_size'],
        shuffle=False, # don't touch if you wanne save probs
        sampler=None,
        pin_memory=True,
        drop_last=False
    )

    ecg_batch, _, labels_batch = next(iter(dataloader))
    print(ecg_batch.shape, labels_batch.shape) # [64, 12000], [64, 13]

    ecg, labels = ecg_batch[0], labels_batch[0]
    print(ecg.shape, labels.shape)

    plt.plot(ecg[:1000])
    plt.show()

    ecg_batch = ecg_batch.to(device) # BS x 12 * L
    labels_batch = labels_batch.squeeze().to(device)
    print(ecg_batch.shape, labels_batch.shape) # [64, 12000], [64, 13]

    with torch.no_grad():
        out = hubert(ecg_batch, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
        
    print(isinstance(hubert, HuBERT)) # False
    print(isinstance(hubert, HuBERTClassification)) # True
        
    if isinstance(hubert, HuBERT):
        logits = hubert.logits(out['last_hidden_state']).transpose(1, 2)
    elif isinstance(hubert, HuBERTClassification):
        logits = out[0]

    print(logits.shape) # [64, 187, 960]

    probs = torch.sigmoid(logits)

    # [BS x num_labels] * N_AUGS
    # probs = torch.sigmoid(logits) if args['task'] == 'multi_label' else torch.softmax(logits, dim=-1)

    # average probs over augmented batches for tta
    # probs = torch.stack(probs).mean(dim=0) if args['tta_aggregation'] == 'mean' else torch.stack(probs).max(dim=0).values

    
    print(probs.shape) # [64, 187, 960]

    threshold = 0.5
    predicted_labels = (probs > threshold).float()
    print(predicted_labels.shape) # [64, 187, 960]


if __name__ == '__main__':
    main()
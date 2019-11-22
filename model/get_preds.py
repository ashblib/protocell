from prototypical_loss import get_query_preds
from query_dataset import QueryDataset, PrototypeDataset, JonQueryDataset, TMPrototypeDataset
from protonet import ProtoNetBig
from parser_pred import get_parser
from misc_utils import fancy_text as fncy_txt

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle as pkl

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def main():
    '''
    Initialize network and write out a map of barcodes to predicted classes
    '''
    options = get_parser().parse_args()
    
    init_seed(options)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initialize dataset and dataloader for the prototype and query datasets
    # Computation of the embeddings can be batched with batch_size to conserve GPU memory
    print("Loading datasets...")
    proto_dataset = PrototypeDataset(root=options.prototypes, ftype=options.proto_ftype, 
                                     cache_path=options.proto_cache_path, use_cached=options.proto_use_cache)
    query_dataset = QueryDataset(root=options.queries, ftype=options.query_ftype, 
                                 cache_path=options.query_cache_path, use_cached=options.query_use_cache)
    query_dataloader = torch.utils.data.DataLoader(query_dataset, batch_size=1024, shuffle=False)
    proto_dataloader = torch.utils.data.DataLoader(proto_dataset, batch_size=1024, shuffle=False)

    # Retrieve misc information from datasets
    proto_class_ids = torch.tensor(proto_dataset.proto_y)  # The class id for each cell in the dataset we're using for prototypes
    class_id_to_labels = proto_dataset.proto_labels        # A dictionary mapping the integer class ids to the actual annotations
    n_features = proto_dataset.proto_x[0].shape[0]         # Number of features (genes) in the data
    query_barcodes = query_dataset.query_bc                # The barcodes for each query cell
    n_query = len(query_barcodes)
    print("")

    # Initialize model
    print("Initializing model...")
    model = ProtoNetBig(x_dim=n_features).to(device)
    model.load_state_dict(torch.load(options.model_path))
    model.eval()
    print("")

    all_proto_embeddings = []  # Prototype cell embeddings for all batches
    all_query_embeddings = []  # Query cell embeddings for all batches

    # First, get embeddings for support cells to be used as prototypes
    print(fncy_txt("Computing prototype embeddings"))
    proto_iter = iter(proto_dataloader)
    for batch in tqdm(proto_iter):
        proto_raw_vectors, _ = batch
        proto_embeddings = model(proto_raw_vectors.to(device))
        all_proto_embeddings.append(proto_embeddings.to('cpu'))
    all_proto_embeddings = torch.cat(all_proto_embeddings, 0)
    print("")
    
    # Then, get embeddings for query cells
    print(fncy_txt("Computing query embeddings"))
    query_iter = iter(query_dataloader)
    for batch in tqdm(query_iter):
        query_raw_vectors, _ = batch
        query_embeddings = model(query_raw_vectors.to(device))
        all_query_embeddings.append(query_embeddings.to('cpu'))
    all_query_embeddings = torch.cat(all_query_embeddings, 0)
    print("")

    # Compute prototype vectors and predictions. This is done on the cpu because of complicated indexing, could probably be sped up
    print(fncy_txt("Computing predictions"))
    predictions, dists = get_query_preds(all_query_embeddings, all_proto_embeddings, proto_class_ids, return_dists=True)
    predictions, dists = predictions.detach().numpy(), dists.detach().numpy()
    print("")

    pred_class_id = [int(x) for x in predictions]
    pred_annotations = [class_id_to_labels[x] for x in pred_class_id]  # Final vector of predicted annotations

    # Write out csv file with the cell barcode and predicted annotation
    with open("./predicted_annotations.csv", "w") as f:
        f.write("barcode,predicted_annotation\n")
        for idx in range(n_query):
            barcode = query_barcodes[idx]
            predicted_annotation = pred_annotations[idx]
            if not idx == (n_query-1):
                f.write("{},{}\n".format(barcode, predicted_annotation))
            else:
                f.write("{},{}".format(barcode, predicted_annotation))

    # Write out csv file with all prototype distances for each cell
    with open("./prototype_distances.csv", "w") as f:
        ordered_labels = [class_id_to_labels[x] for x in range(len(list(class_id_to_labels.keys())))]
        f.write("barcode")
        for label in ordered_labels:
            f.write(",{}".format(label))
        f.write("\n")
        for idx in range(n_query):
            barcode = query_barcodes[idx]
            distances = dists[idx]
            f.write(barcode)
            for dist in distances:
                f.write(",{}".format(round(dist, 3)))
            f.write("\n")

    print("Done! o/~")

if __name__ == '__main__':
    main()
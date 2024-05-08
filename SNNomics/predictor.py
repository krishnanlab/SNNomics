import torch
import torch.nn as nn
import pandas as pd


class Predictor:
    def __init__(self, query, model, criterion, predict_loader, device, outdir):
        self.query = query
        self.model = model
        self.criterion = criterion
        self.predict_loader = predict_loader
        self.device = device
        self.outdir = outdir
        self.pred_dict = {'sample': [], 'similarity': []}

    def predict(self):
        with torch.no_grad():
            for i, data in enumerate(predict_loader):
                _id, sample = data
                sample.to(device)
                input_query = self.query.to(device)
                
                output_query, output_sample = self.model(input_query, sample)
                similarity = criterion(output_query, output_sample) 
                
                self.pred_dict['sample'].append(_id)
                self.pred_dict['similarity'].append(similarity.item())

    def save_predictions(self, outfile) 
        df = pd.DataFrame.from_dict(self.pred_dict)
        df.to_parquet(self.outdir / outfile)        


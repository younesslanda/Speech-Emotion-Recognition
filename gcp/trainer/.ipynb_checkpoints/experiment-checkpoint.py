#Author : Youness Landa

from config import Config as cfg

class Experiment:
    def __init__(self, dataloader_train, optimizer, loss):
        self.dataloader_train = dataloader_train
        self.optimizer = optimizer
        self.loss = loss
        
    def run(self, model):
        for epoch in range(cfg.NUM_EPOCHS):
            for i_batch, sample_batched in enumerate(self.dataloader_train):
                features, lengths, emotion_idxs, gender_idxs = sample_batched
                
                self.optimizer.zero_grad()
                
                emotion_predictions, gender_prediction = model(features, lengths)
                
                emotion_loss = self.loss(emotion_predictions,emotion_idxs)
                gender_loss = loss( pred_gender,labels_gen.squeeze())
                total_loss = args.alpha*emotion_loss+args.beta*gender_loss
                total_loss.backward()
                optimizer.step()
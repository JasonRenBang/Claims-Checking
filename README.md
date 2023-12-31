# Claims-Checking

The RoBerTa and BERT models have been compared with fscore and accuracy, and finally I use RoBerTa model to create sample that contains positive and negative evidences and than I still use RoberTa model to continue the training and find  out the best five evidence that may important to analyse a claim and label it. And than I construct the trained BERT model to decide the label with the claims by supports, refutes, not_enough_info, disputed.

import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

MAX_SEQ_LENGTH = 512
labels = [{"x":64.28711265942997,"y":28.966669710882663,"width":13.503177643707845,"height":1.1924884152884836,"rotation":0,"labels":["Returns and allowances"],"original_width":2550,"original_height":3300},
          {"x":82.09167024442038,"y":30.60634128190431,"width":16.045664357481744,"height":1.0434273633774245,"rotation":0,"labels":["Balance. Substract line 1b from line 1a"],"original_width":2550,"original_height":3300},
          {"x":81.61978631965741,"y":32.24601285292598,"width":15.392462469218064,"height":0.9316315744441334,"rotation":0,"labels":["Cost of goods sold"],"original_width":2550,"original_height":3300},
          {"x":83.19964768197075,"y":34.063995687284745,"width":14.61169205310524,"height":1.0434273633774485,"rotation":0,"labels":["Gross profit. Substract line 2 from line 1c"],"original_width":2550,"original_height":3300},
          {"x":81.84055174718519,"y":34.9816791121002,"width":15.35922655138036,"height":1.6121690542508762,"rotation":0,"labels":["Net gain (loss) from Form 4797, line 17"],"original_width":2550,"original_height":3300},
          {"x":82.33094740315646,"y":36.73790484656557,"width":15.26562022404282,"height":1.3797696488145983,"rotation":0,"labels":["Other income (loss)"],"original_width":2550,"original_height":3300},
          {"x":82.11170544918113,"y":38.26073309688247,"width":16.923639390246066,"height":1.3262686960377001,"rotation":0,"labels":["Total income (loss). Add lines 3 through 5"],"original_width":2550,"original_height":3300},
          {"x":82.44921384854264,"y":39.4776308739628,"width":15.982747006681342,"height":1.5769964918377095,"rotation":0,"labels":["Compensation of officers"],"original_width":2550,"original_height":3300},
          {"x":82.9885014327046,"y":41.200552866644834,"width":15.779896244697387,"height":1.4493335005001562,"rotation":0,"labels":["Salaries and wages (less employment credits)"],"original_width":2550,"original_height":3300},
          {"x":83.09830781627717,"y":43.15068493150684,"width":14.403706688154625,"height":1.0273972602739465,"rotation":0,"labels":["Repairs and maintenance"],"original_width":2550,"original_height":3300},
          {"x":83.31990330378729,"y":43.83561643835616,"width":12.85253827558416,"height":1.7123287671232712,"rotation":0,"labels":["Bad debts"],"original_width":2550,"original_height":3300},
          {"x":83.41098217636105,"y":45.21372568967203,"width":13.034099796960987,"height":2.054794520547948,"rotation":0,"labels":["Rents"],"original_width":2550,"original_height":3300},
          {"x":83.31990330378729,"y":47.43150684931507,"width":13.295729250604355,"height":1.0273972602739818,"rotation":0,"labels":["Taxes and licenses"],"original_width":2550,"original_height":3300},
          {"x":81.99033037872684,"y":48.8013698630137,"width":14.182111200644643,"height":1.0273972602739647,"rotation":0,"labels":["Interest (see instruction)"],"original_width":2550,"original_height":3300},
          {"x":82.71555197421434,"y":50.513698630137014,"width":15.954875100725122,"height":1.0273972602739325,"rotation":0,"labels":["Depreciation not claimed on Form 1125-A or elsewhere on return"],"original_width":2550,"original_height":3300},
          {"x":82.84741044612117,"y":52.11281557794627,"width":17.152589553878716,"height":0.9906034189969515,"rotation":0,"labels":["Depletion (Do not deduct oil and gas depletion.)"],"original_width":2550,"original_height":3300},
          {"x":83.31990330378736,"y":53.42465753424644,"width":15.290088638194938,"height":1.198630136986419,"rotation":0,"labels":["Advertising"],"original_width":2550,"original_height":3300},
          {"x":81.32554391619662,"y":55.13698630136985,"width":16.61966156325554,"height":1.0273972602739747,"rotation":0,"labels":["Pension, profit-sharing, etc., plans"],"original_width":2550,"original_height":3300},
          {"x":82.65511684125705,"y":56.33561643835616,"width":16.17647058823529,"height":1.1986301369863028,"rotation":0,"labels":["Employee benefit programs"],"original_width":2550,"original_height":3300},
          {"x":82.2119258662369,"y":57.70547945205481,"width":16.8412570507655,"height":1.1986301369862842,"rotation":0,"labels":["Other deductions (attach statement)"],"original_width":2550,"original_height":3300},
          {"x":82.65511684125707,"y":59.24657534246581,"width":17.062852538275585,"height":1.3698630136985743,"rotation":0,"labels":["Total deductions. Add lines 7 through 19"],"original_width":2550,"original_height":3300},
          {"x":83.82352941176467,"y":60.958904109589,"width":15.290088638194979,"height":1.027397260273859,"rotation":0,"labels":["Ordinary business income (loss). Substract line 20 from line 6"],"original_width":2550,"original_height":3300},
          {"x":65.3706688154714,"y":27.56849315068493,"width":12.40934730056405,"height":1.1986301369863028,"rotation":0,"labels":["Gross receipts or sales"],"original_width":2550,"original_height":3300}]



class LayoutLMTokenizerWithInputShape(LayoutLMTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_names = self.model_input_names + ["input_shape"]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        else:
            return self.vocab["[UNK]"]

    def encode_plus(self, *args, **kwargs) -> dict:
        inputs = super().encode_plus(*args, **kwargs)
        inputs["input_shape"] = torch.tensor([1, self.model_max_length], dtype=torch.long)
        return inputs

#define PyTorch dataset for form parsing
class FormDataset(Dataset):
    def __init__(self, labels: list, tokenizer: LayoutLMTokenizer, max_seq_length: int):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

#return total number of samples in the dataset
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]

        # Extract text from the bounding box
        text = " ".join(label['labels'])  # Concatenate all labels into one text

        # Tokenize the text
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_seq_length)

        # Adding coordinates and other information as input features
        coords = torch.tensor([label['x'], label['y'], label['width'], label['height']], dtype=torch.float32)
        rotation = torch.tensor(label['rotation'], dtype=torch.float32)
        original_width = torch.tensor(label['original_width'], dtype=torch.float32)
        original_height = torch.tensor(label['original_height'], dtype=torch.float32)

        # Map the original label to the corresponding index in the tokenizer vocabulary
        tokens = self.tokenizer.tokenize(text)
        labels = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

        # Pad the input sequences and labels together to the same length
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = torch.nn.functional.pad(labels, (0, self.max_seq_length - labels.size(0)), "constant", -100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'coords': coords,
            'rotation': rotation,
            'original_width': original_width,
            'original_height': original_height
        }

def compute_metrics(pred: tuple) -> dict:
    logits, labels = pred
    predictions = np.argmax(logits,axis=-1)

    # Flatten the labels and predictions
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()

    # Compute the F1 score
    f1 = f1_score(labels_flat, predictions_flat, average='weighted', zero_division=1)
    precision, recall, _, _ = precision_recall_fscore_support(labels_flat, predictions_flat, average='weighted',
                                                              zero_division=1)

    return {
        'eval_f1_score': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }

# Initialize LayoutLM tokenizer and model
tokenizer = LayoutLMTokenizerWithInputShape.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,  # Set evaluation batch size
    logging_dir="./logs",
)

# Split data into training and evaluation sets
train_labels, eval_labels = train_test_split(labels, test_size=0.1, random_state=42)

# Create datasets
train_dataset = FormDataset(labels=train_labels, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
eval_dataset = FormDataset(labels=eval_labels, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()


eval_results = trainer.evaluate(eval_dataset)

train_results = trainer.evaluate(train_dataset)

print("Training Set Metrics:")
print("F1-score:", train_results['eval_f1_score'])
print("Precision:", train_results['eval_precision'])
print("Recall:", train_results['eval_recall'])

print("Evaluation Set Metrics:")
print("F1-score:", eval_results['eval_f1_score'])
print("Precision:", eval_results['eval_precision'])
print("Recall:", eval_results['eval_recall'])






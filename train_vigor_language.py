from transformers import Trainer, TrainingArguments, \
                         CLIPVisionConfig, CLIPTokenizerFast                      

from sample4geo.dataset.vigor_plus import VigorCombinedDataset
from torchvision import transforms
import transformers
import torch
text_processor = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=348, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
transform_val = transforms.Compose([
        transforms.Resize((348, 348)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
dataset_root = '/home/erzurumlu.1/yunus/research_drive/data/VIGOR'
cities_to_include_train = ['NewYork', 'Chicago']
cities_to_include_val = ['Seattle', 'SanFrancisco']
train_dataset = VigorCombinedDataset(
    dataset_root,
    cities_to_include_train,
    transform=transform_train
)
val_dataset = VigorCombinedDataset(
    dataset_root,
    cities_to_include_val,
    transform=transform_val,
    val=True
)
def collate_fn(examples):
    images = [example[0] for example in examples]
    text = [example[1] for example in examples]
    text = text_processor(text, return_tensors="pt", padding=True, truncation=True)
    # text['interpolate_pos_encoding'] = True
    text['pixel_values'] = torch.stack(images)
    # text['return_loss'] = True
    return text

train_args = TrainingArguments(
        output_dir='/home/erzurumlu.1/yunus/research_drive/language_pretrain',
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8, # 8 for 4 GPUs
        learning_rate=1e-06,
        weight_decay=0.001,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        max_grad_norm=1.0,
        num_train_epochs=20,
        max_steps=-1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.2,
        logging_first_step = False,
        logging_steps=1,
        save_strategy='steps',
        save_steps=50,
        seed=42,
        dataloader_drop_last=True,
        run_name=None,
        adafactor=False,
        report_to='tensorboard',
        skip_memory_metrics=True,
        resume_from_checkpoint=None,
    )
vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16", hidden_act = "gelu")

class CLIP(transformers.models.clip.modeling_clip.CLIPModel):
    def forward(self, **kwargs):
        return super().forward(**kwargs, interpolate_pos_encoding=True, return_loss=True)
model = CLIP.from_pretrained("openai/clip-vit-base-patch16",vision_config=vision_config)
trainer = Trainer(model=model,
                  args=train_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=collate_fn,
)

trainer.train()

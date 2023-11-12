# GPT-related metrics

This directory contains scripts to evaluate GPT-related metrics.

First, finetune GPT-2 on covertext and stegotext, seperately. We then have 3 GPT-2 models:
* Vanilla pratraied GPT-2
* GPT-2 finetuned on covertext
* GPT-2 finetuned on stegotext

## PPL

Two types of PPL of stegotext are evaluated:
* `PPL-normal`: PPL of stegotext evaluated on the vanilla pretrained GPT-2 model
* `PPL-cover`: PPL of stegotext evaluated on the model finetuned on covertext

## JSD

JSD should only be evaluated on the **same** test set.

Two types of JSD of stegotext are evaluated:
* `JSD-normal`: JSD of stegotext evaluated on the vanilla pretrained GPT-2 model
* `JSD-cover`: JSD of stegotext evaluated on the model finetuned on covertext

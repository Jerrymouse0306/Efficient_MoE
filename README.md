# Memory Efficient MoE Model Fine-tuning based on Hybrid Optimization

### Installation
```bash
conda create --name loho python=3.9.18
conda activate loho
pip install -r requirements.txt
```

### RoBERTa-large experiments
```bash
cd medium_models

#run inter-layer hybrid optimization with sgd
bash loho_sgd_inter.sh

#run inter-layer hybrid optimization with Adam
bash loho_adam_inter.sh

#run intra-layer hybrid optimization with Adam
bash loho_adam_intra.sh
```
You can modify the task name and hyperparamters in these scripts.

### OPT-13B experiments
```bash
cd large_models

#run inter-layer hybrid optimization with sgd
bash loho_sgd_inter.sh

#run inter-layer hybrid optimization with Adam
bash loho_adam_inter.sh

#run intra-layer hybrid optimization with Adam
bash loho_adam_intra.sh
```
You can modify the task name and hyperparamters in these scripts.

### Acknowledgment
The implementation of our method is built upon the foundation laid by [MeZO](https://github.com/princeton-nlp/MeZO)

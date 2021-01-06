# Transformer-Grapheme-to-Phoneme-Conversion
 Transformer based model for Grapheme to Phoneme Conversion.<br/>
 Instead of directly adding inputs to positional encodings, LayerNormalisation is done on inputs to improve diagonal attention which is a crucial factor in Grapheme to Phoneme task.<br/>
 
 ### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Bheshaj-Kumar/Transformer-Grapheme-to-Phoneme-Conversion.git
   ```
2. Install Following Packages
   * tensorflow==2.3
   * ruamel.yaml
   * editdistance
   * tqdm
   * bunchify

<!-- USAGE EXAMPLES -->
## Usage
  #### Training

   ```sh
   python train_g2p.py --config_path /path/to/model_config.yaml --session_name "session_name"  --data_dir /path/to/data -sv_file /path/to/source_vocab_file -tv_file /path/to/target_vocab_file 
   ```
   
  #### Inference
   
   ```sh
   python train_g2p.py --config_path /path/to/model_config.yaml --session_name "session_name"  --data_dir /path/to/data --eval -sv_file /path/to/source_vocab_file -tv_file /path/to/target_vocab_file    
   ```

## Note
  * This repo already contains the required CMU DICT data at their default paths. If you want to change data, change the path accordingly.

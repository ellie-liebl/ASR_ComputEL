J. Elizabeth Liebl; jliebl@gmu.edu
03/10/2026
ASR for ComputEL
README

Follow these steps to train the models:
1. Prepare a mdc_ids.tsv: you need a MDC API key, download path, dataset ID, and Vox Communis pipeline tag

2. Run download_data.py in an environment running python 3.11+

3. Run download_textgrids.py in an environment running python 3.11+

4. Unzip downloads using unzip_data.py

5. Run gen_transcripts.py to create DFs of audio path + transcript pairs. 

6. Run set_splits.py to create test / train DFs. This is done before processing the transcripts to prevent dealing with millions of rows at once. 

7. Run preprocess_transcripts.py to clean diacritics and replace multi-character affricates with ligatures. 

8. Run create_datasets.py to create HF/Torch compatible datasets. 

9. Create model vocabularies (vocab.json) with vocab_builder.py. 

10. Run main_gpt_edit.py to train each model in python 3.9.9(!! It must be this exact version!!). 

11. Evaluate your models with all_model_eval.py

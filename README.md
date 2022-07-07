# amex_default_kaggle
Code for American Express' Default Prediction competition, hosted on Kaggle

# Experiments and Ideas Log

|Idea |Description | Finding|
--- | --- | ---|
|Base FE |Aggregates, last rel mean| solid .797 baseline with dart |
|NA FE |Derive feature signal from NA stats and clusters| * |
|Flatten Risk Score FE |Get all customer risk ratings not just last and stats| model trained on full flattened dataset gives ensemble boost |
|Data aug: multiple statements| Use multiple (e.g. 2nd last) statements for training data| 2 statements does improve a default feature model |
|Downsampling for model diversity| E.g. sandwich technique| `scale_pos_weight` did not add to ensemble |
|Excl. internal score for diversity| Model w/o super feature in ensemble| * |
|Difference/trend features| Diffs of various period| * |
|Matrix factorization cats | Apply to full category history | * |

# FROZEN - the state of the project before the 2025 test

Written **2026-10-16 22:51:20** at commit `e4c53b8209c73daf281f77a1aefb73454188bb60`.

Everything below was decided using 2023 and 2024 only. After this file was committed, no setting, threshold or model choice was changed. The 2025 test is scored once against exactly this.

## The split

- train **[2023]**, validation **[2024]**, test **[2025]** (hold-out)
- 30 stations, horizons [6, 12, 24, 48] h, classes Low < 5.0 mg/L, Medium to 7.0, Good above
- seed 42 everywhere

## The final system

`final_system = stacking` - chosen in step 7 by a rule written before the result was known. The context-aware ensemble (CAE) and the adaptive threshold-aware ensemble (ATAE) were both tested and **not adopted**; they are still scored on 2025 for the record.

## Members and settings

- **random_forest**: `{'n_estimators': 300, 'min_samples_leaf': 20, 'max_features': 0.3, 'max_samples': 0.5, 'n_jobs': -1, 'classifier_class_weight': 'balanced_subsample'}`
- **xgboost**: `{'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.7, 'n_jobs': -1}`
- **lightgbm**: `{'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'n_jobs': -1, 'categorical_features': ['station_id'], 'classifier_class_weight': 'balanced'}`
- **bilstm / cnn_lstm**: variant `reg_hybrid_72h_small` (`{'dropout': 0.3, 'optimizer': 'adamw', 'weight_decay': 0.0001, 'learning_rate': 0.0005, 'early_stopping_patience': 4, 'use_engineered_features': True, 'feature_hidden': 64, 'sequence_length': 72, 'hidden_size': 32}`), sequence length 72 h, seeds [42, 43, 44], predictions averaged

## Rounds and epochs to use when retraining (no early stopping)

The 2025 year must not be watched, so the number of rounds is fixed here at whatever step 3 and step 5 found on 2024.

| model / task / horizon | rounds |
|---|---|
| lightgbm_classification_h12 | 284 |
| lightgbm_classification_h24 | 203 |
| lightgbm_classification_h48 | 184 |
| lightgbm_classification_h6 | 363 |
| lightgbm_regression_h12 | 394 |
| lightgbm_regression_h24 | 252 |
| lightgbm_regression_h48 | 199 |
| lightgbm_regression_h6 | 1052 |
| xgboost_classification_h12 | 396 |
| xgboost_classification_h24 | 226 |
| xgboost_classification_h48 | 173 |
| xgboost_classification_h6 | 446 |
| xgboost_regression_h12 | 555 |
| xgboost_regression_h24 | 182 |
| xgboost_regression_h48 | 177 |
| xgboost_regression_h6 | 850 |

| deep model | seed | epochs |
|---|---|---|
| bilstm | 42 | 3 |
| bilstm | 43 | 3 |
| bilstm | 44 | 2 |
| cnn_lstm | 42 | 4 |
| cnn_lstm | 43 | 4 |
| cnn_lstm | 44 | 4 |

## Warning thresholds

Chosen on the OUT-OF-FOLD 2024 probabilities from step 7, as the threshold catching the most onsets with a false-alarm rate at or below **1.8%** (target 2.0% minus a 0.2% safety margin, because step 7 showed a tau tuned to the limit overshoots out of sample).

| method | horizon | tau | onset recall (2024 oof) | false alarm (2024 oof) |
|---|---|---|---|---|
| persistence | 6 h | argmax | - | - |
| persistence | 12 h | argmax | - | - |
| persistence | 24 h | argmax | - | - |
| persistence | 48 h | argmax | - | - |
| linear | 6 h | 0.46 | 0.835 | 0.0178 |
| linear | 12 h | 0.45 | 0.815 | 0.0176 |
| linear | 24 h | 0.45 | 0.389 | 0.0174 |
| linear | 48 h | 0.50 | 0.364 | 0.0173 |
| random_forest | 6 h | 0.42 | 0.833 | 0.0176 |
| random_forest | 12 h | 0.36 | 0.835 | 0.0178 |
| random_forest | 24 h | 0.35 | 0.503 | 0.0172 |
| random_forest | 48 h | 0.41 | 0.460 | 0.0179 |
| xgboost | 6 h | 0.31 | 0.871 | 0.0178 |
| xgboost | 12 h | 0.28 | 0.855 | 0.0176 |
| xgboost | 24 h | 0.30 | 0.531 | 0.0176 |
| xgboost | 48 h | 0.39 | 0.502 | 0.0175 |
| lightgbm | 6 h | 0.31 | 0.869 | 0.0177 |
| lightgbm | 12 h | 0.31 | 0.847 | 0.0174 |
| lightgbm | 24 h | 0.31 | 0.521 | 0.0180 |
| lightgbm | 48 h | 0.44 | 0.484 | 0.0177 |
| bilstm | 6 h | 0.44 | 0.826 | 0.0179 |
| bilstm | 12 h | 0.42 | 0.799 | 0.0179 |
| bilstm | 24 h | 0.43 | 0.461 | 0.0177 |
| bilstm | 48 h | 0.45 | 0.437 | 0.0174 |
| cnn_lstm | 6 h | 0.44 | 0.834 | 0.0180 |
| cnn_lstm | 12 h | 0.39 | 0.805 | 0.0176 |
| cnn_lstm | 24 h | 0.41 | 0.484 | 0.0179 |
| cnn_lstm | 48 h | 0.45 | 0.468 | 0.0177 |
| simple_average | 6 h | 0.36 | 0.871 | 0.0178 |
| simple_average | 12 h | 0.33 | 0.858 | 0.0179 |
| simple_average | 24 h | 0.34 | 0.529 | 0.0176 |
| simple_average | 48 h | 0.41 | 0.498 | 0.0174 |
| stacking | 6 h | 0.27 | 0.871 | 0.0179 |
| stacking | 12 h | 0.24 | 0.857 | 0.0177 |
| stacking | 24 h | 0.24 | 0.529 | 0.0174 |
| stacking | 48 h | 0.35 | 0.494 | 0.0176 |
| atae | 6 h | 0.35 | 0.874 | 0.0177 |
| atae | 12 h | 0.32 | 0.859 | 0.0178 |
| atae | 24 h | 0.34 | 0.525 | 0.0172 |
| atae | 48 h | 0.41 | 0.494 | 0.0172 |
| cae | 6 h | 0.37 | 0.875 | 0.0176 |
| cae | 12 h | 0.39 | 0.846 | 0.0174 |
| cae | 24 h | 0.34 | 0.510 | 0.0173 |
| cae | 48 h | 0.39 | 0.494 | 0.0174 |

## Ablation thresholds

| input set | horizon | tau | onset recall (2024) |
|---|---|---|---|
| W | 6 h | 0.32 | 0.867 |
| W | 12 h | 0.30 | 0.851 |
| W | 24 h | 0.32 | 0.507 |
| W | 48 h | 0.44 | 0.473 |
| W+M | 6 h | 0.31 | 0.868 |
| W+M | 12 h | 0.31 | 0.851 |
| W+M | 24 h | 0.33 | 0.515 |
| W+M | 48 h | 0.44 | 0.488 |
| W+M+A | 6 h | 0.31 | 0.870 |
| W+M+A | 12 h | 0.31 | 0.852 |
| W+M+A | 24 h | 0.32 | 0.519 |
| W+M+A | 48 h | 0.43 | 0.491 |

## Pass criteria (unchanged since step 3)

```
{
  "regression": {
    "min_skill_vs_persistence": 0.0
  },
  "classification": {
    "beat_persistence_on": "f1_macro",
    "not_worse_on": "recall_Low",
    "recall_tolerance": 0.0
  },
  "min_horizons_passed": 3
}
```

## File hashes

SHA-256 of every model file the 2025 test reads. If one of these changes, the test is no longer the one described here.

| file | sha256 | bytes |
|---|---|---|
| models/dl/bilstm.pt | `3ff98d1ac9425fdda712384103262999cc58c695159fb681a476d928b9729ac0` | 645,556 |
| models/dl/bilstm_history.csv | `bd80ef78d0927f2dff2ca46e6023fb9e8d2ded0d5999121db71f04db9c51c7e7` | 945 |
| models/dl/bilstm_seed42.pt | `3bec1ebe8b821630e39bf9a1f9f0fe58f222d2ee0e3fe323c80a0f9938308e33` | 269,900 |
| models/dl/bilstm_seed43.pt | `6e82669e3fecd2b8e01e88d9948da03be9ff350780fdeab50f02aa5b461443de` | 269,900 |
| models/dl/bilstm_seed44.pt | `9456ea9ff4c6f074ce53e8a84fb1cda192e622122d76332671c366ac5e2bf80e` | 269,900 |
| models/dl/bilstm_seed_spread.csv | `135f802cc5e74d66f7d8031204fd1b4dca307137a9ea582ac1de361786c4b08a` | 451 |
| models/dl/cls_fix_summary.json | `3d2deb6ea1765fcaf6aa745a96f63995885b5ae28bbfbc91ba731ee258f4b933` | 1,075 |
| models/dl/cnn_lstm.pt | `469e73ca38f4f33662921f4cbcfb70a70e58b8c953ed4b43b478e56da2bc57f3` | 200,386 |
| models/dl/cnn_lstm_history.csv | `d8d315cbd5b9ccdfe7c692b80f12c4f3973fe5d2d34f30ddec281201c4fbfb8b` | 942 |
| models/dl/cnn_lstm_seed42.pt | `3def2fc00e876c32e9bbcc82d5fd4c3077537e052f3207f328c281444f7bad2f` | 180,668 |
| models/dl/cnn_lstm_seed43.pt | `aae4be0656e03d470e3a70407de305e827a09166a0a16f5e350273c9931a535c` | 180,668 |
| models/dl/cnn_lstm_seed44.pt | `8b89407da8c011cfa47f141eb5bc3fa334d5f89783297f2fed2ad27fec8f2d16` | 180,668 |
| models/dl/cnn_lstm_seed_spread.csv | `aeecf6d6227268fbdf89d052dc6f84280a35589692c6923849f1faac208bfc61` | 454 |
| models/dl/scaler.joblib | `1e83595e27a86389ae9054eccda5e86593b7df3b65d41f2f20112afc8c2142d6` | 25,645 |
| models/dl/training_summary.json | `0ee0aab97030eb34bac797d1d0348033f38139a19f23285adf91350aed6a67e4` | 2,697 |
| models/ml/lightgbm_classification_h12.txt | `f78b47ecb5f97c02262220fdab8a6d953a9170fd47550eaf75806678d29adce2` | 5,958,007 |
| models/ml/lightgbm_classification_h24.txt | `fce49409757b1caae450493542a222c11a7d9ac847be39e1c65490132d002548` | 4,258,287 |
| models/ml/lightgbm_classification_h48.txt | `cd31e9e79ccbc5ba0e6b2f5cf2fb3f0bccf4d9d1ae556e0ede58184d56f127a2` | 3,867,732 |
| models/ml/lightgbm_classification_h6.txt | `4d2a4f726d3a358fc3accf95fb949930906685d2d790e85d8b23f26632642027` | 7,623,874 |
| models/ml/lightgbm_regression_h12.txt | `3ca5148e6326a52732ee78eede6e6072e27e53418f1fd1cbcc8b5b56bb89ad3f` | 2,327,075 |
| models/ml/lightgbm_regression_h24.txt | `7fce49c83a139fcfbbe5a7157dd83b7952bee27d2f1a6ffa0077714287de8fbc` | 1,493,840 |
| models/ml/lightgbm_regression_h48.txt | `4b623a2a7ec9338a27c0cf839300abe80f3abb8efcaf6ddd06b5194d8937e95c` | 1,188,078 |
| models/ml/lightgbm_regression_h6.txt | `fd5ec89d1ba87ae2f27f2fe9e20607e1e570520bc0b418261b753a46454616ba` | 6,186,601 |
| models/ml/linear_classification_h12.joblib | `a38ae94569690718163e76f04c6c0297ad262100a2958467d79615c92203b665` | 12,782 |
| models/ml/linear_classification_h24.joblib | `69563207e6143f2eaafa2269b5fa8e778092db65a6e8f95c104d3c07a900704f` | 12,761 |
| models/ml/linear_classification_h48.joblib | `10f05cabc8619ab80a3dc19da88c278e69ce07e36e938da2b78385d5336fa37d` | 12,753 |
| models/ml/linear_classification_h6.joblib | `25e11f6af800cb2611b8cd9a17aae4aee3cff706c2be4f8e8417ef34c8d69a9a` | 12,772 |
| models/ml/linear_regression_h12.joblib | `e53b24508061984e75fda1b9e1afe1ceee4337bc76c501dce139bb49c93e7027` | 9,189 |
| models/ml/linear_regression_h24.joblib | `5bee948aaeed2934c3763b61c1825d96514a7f691e9613f2b295abd3a8077881` | 9,233 |
| models/ml/linear_regression_h48.joblib | `7d8270aa4ba84bff65d3a72c385c02440b8365fe38545cc93873c7abcf40e71e` | 9,227 |
| models/ml/linear_regression_h6.joblib | `22648bf29dc4508b8e2f8e4b24168591c91a706850e22cf8f874245603874e27` | 9,238 |
| models/ml/random_forest_classification_h12.joblib | `64350f6c8a72ef0cd30c257e2a08d9309061c853a98b1a389dcbf1eccc2508d1` | 34,517,316 |
| models/ml/random_forest_classification_h24.joblib | `02bc58edc092cc7feb3a2d2d3b7e2438fd2845b0fc4b25aee7950c7bea33d489` | 34,940,003 |
| models/ml/random_forest_classification_h48.joblib | `8cc92d9a4e04e6d5590d3caa50e5e1192130ecc593ac58672315fdfbc3cdb8fa` | 41,905,175 |
| models/ml/random_forest_classification_h6.joblib | `2b56b474a494ac98a894830eeb5bb6a966701de784b1d7439650743ad79cb325` | 35,293,855 |
| models/ml/random_forest_regression_h12.joblib | `8a5f54be62187a77bd4eb5e89d8e8fb634d23f9cfee0a0283acd7526329b18f3` | 58,692,339 |
| models/ml/random_forest_regression_h24.joblib | `883d916d32ad67dd347b5e4da24973894523a0a8d160871715ba237e36dc03c6` | 58,846,270 |
| models/ml/random_forest_regression_h48.joblib | `2eb9f2bf97928ed4d71312bf582ea68ad1f9e5695a4b3949e8401ad4beb968bf` | 58,895,988 |
| models/ml/random_forest_regression_h6.joblib | `ab197c48aef9da5f883e28e1edc21b7d10314fdf0aceac6dbf718924c43f6fd9` | 58,655,605 |
| models/ml/xgboost_classification_h12.json | `31617efce94cdc3007f36643161c9eb19a2838aeffffe736b1e039a6abba1c47` | 25,313,953 |
| models/ml/xgboost_classification_h24.json | `69ab67582dd0a80fb7f24c3a4eb19a8dba84fa1686ddd0310a1378862272d220` | 16,649,579 |
| models/ml/xgboost_classification_h48.json | `9db9c5079f3fde6f2e0e957d979bb7dfa6d5425c3b018edbd3a45da575c7c481` | 14,396,484 |
| models/ml/xgboost_classification_h6.json | `17b8e31661774466839de8477d04e916e05350e4faa5764390cdb6980fb06c81` | 28,382,346 |
| models/ml/xgboost_regression_h12.json | `fdf3704c8e27ebb864570508f54f2850c8ee1fc970eab1836d6b2fac60ec3ee0` | 13,443,619 |
| models/ml/xgboost_regression_h24.json | `5c236da54789ead91e4fb4d77b81946b04599354ed9fa663ef712d61a958459b` | 5,456,704 |
| models/ml/xgboost_regression_h48.json | `0947624a15bb5ce429c171a622047d902488babca8892da64cac7387f2faf8ea` | 5,525,276 |
| models/ml/xgboost_regression_h6.json | `0de0805e37b1b3dd0ac8a6793068ef68cc715d3f7e1eaf9e9efa3d07307eff14` | 20,143,409 |
| models/ensemble/ablation_thresholds.json | `2a03ae87658210e7f2672c3c4a5eb69780a5324dcdba37af3f7642379730059b` | 2,105 |
| models/ensemble/cae_classification_h12_final.joblib | `3b3895dfddf466d090f1728606c5d1ee2dacc5d44826cd4a80153a1af77300d4` | 1,600,971 |
| models/ensemble/cae_classification_h24_final.joblib | `fa9c9dd605323d63006c35b9396334bdee4c0ed9d3535c2dda08b22e38aef1bb` | 1,483,467 |
| models/ensemble/cae_classification_h48_final.joblib | `bb79b4c57724fd8f51a6a5243c48a24287048d02a5439adc078eabdd2373504a` | 1,393,019 |
| models/ensemble/cae_classification_h6_final.joblib | `662ca1c33cd623c359f075087d950d1f0b75ecfa7709b7f3de651acbc684a701` | 1,231,451 |
| models/ensemble/cae_regression_h12_final.joblib | `d6dec8fcf415783d283c78f8c78a6182c2adea271db53a687e7f83ff22e4c47f` | 265,854 |
| models/ensemble/cae_regression_h24_final.joblib | `1134ab5d51975d9d08a94264941be41dfb3a89d40f9365e440c0d9f88fb3da10` | 206,767 |
| models/ensemble/cae_regression_h48_final.joblib | `ec1dad200fe2bec835540f67a8ec57c6ee4cb33b0fa56b295d693a5f12c7a7e5` | 211,548 |
| models/ensemble/cae_regression_h6_final.joblib | `3b088d3bc2619483ba70ef4a2e9607b03ae167ab641b9b87b56706e710e635a4` | 377,010 |
| models/ensemble/stacking_classification_h12_final.joblib | `21f7ad8606bd9490f8cdb71025a7eba06a0b5b186aad297cc801fa7f43dff6b9` | 1,355 |
| models/ensemble/stacking_classification_h24_final.joblib | `6ded599029a0aa63fff20a7b9afcae420e69cce8ed2a2320827a4c1e69621521` | 1,355 |
| models/ensemble/stacking_classification_h48_final.joblib | `c5ae9cd5ac4f455304d4e06355daafeb5fbdb041b897b1c65d7c94cc6a92e1bc` | 1,355 |
| models/ensemble/stacking_classification_h6_final.joblib | `403341dc37a71f21cd1582436a45d45cc5584b8aa4f42d7d317ff458236348a2` | 1,355 |
| models/ensemble/stacking_regression_h12_final.joblib | `b94fdcd935f34fd1f7df89171152c54dccd2be64e357ffe441964ead4c976dc4` | 691 |
| models/ensemble/stacking_regression_h24_final.joblib | `b8840dd1f958b43da1ea99ca8f32bea7cb6dac28b0a3dbfb013135ecbf6fba56` | 691 |
| models/ensemble/stacking_regression_h48_final.joblib | `3642b1e9284e8d1e8e9cd229b95cc0ba4e590771652120102ed7b28a016b1cef` | 691 |
| models/ensemble/stacking_regression_h6_final.joblib | `233a3f78167e3cd07227712e1c17b1f54e90495f01699aa9e5727f5944451b76` | 691 |
| models/ensemble/warning_thresholds.json | `91dbfa5e6ac3e85e17ee9487e1bbd5e7083460e62b568f878bd23918b1afad0c` | 7,683 |
| models/ablation/ablation_W+M+A_classification_h12_2023.joblib | `e0ed761d7bf30a0ff4c67058c98c29affce1e3f2402777d53a95bd6b684a1a6f` | 5,966,531 |
| models/ablation/ablation_W+M+A_classification_h24_2023.joblib | `a2c862796a4d1cc97963fd4e62f74d61def1b2a34516ee8c4049b689dc18f9a6` | 4,265,411 |
| models/ablation/ablation_W+M+A_classification_h48_2023.joblib | `7f7cdd229eff1396129fdcb151b8d68b5ea2e7d637006621965b6a1b68df9654` | 3,875,427 |
| models/ablation/ablation_W+M+A_classification_h6_2023.joblib | `574f28c5f9d51c442750e9b35c9532d8895009e8a0e051fa0063076262e5b2fe` | 7,632,739 |
| models/ablation/ablation_W+M_classification_h12_2023.joblib | `edca6a256fa2b9868797c2b678e1cb3ea20a22a529e8244c055ec1f315415427` | 5,964,775 |
| models/ablation/ablation_W+M_classification_h24_2023.joblib | `474658a99e60ccad65eaf570ae28095aed3b5f719351a18e2c12fbcf7db53ba5` | 4,262,775 |
| models/ablation/ablation_W+M_classification_h48_2023.joblib | `eb95b151347c50871ac9da17c5055f4f3f7dff237779ef52e66cca8542664153` | 3,874,007 |
| models/ablation/ablation_W+M_classification_h6_2023.joblib | `8faff38da10147d1b89a3c53572f248c1c5a49d0525b8e93486286d2f86c4ff9` | 7,631,431 |
| models/ablation/ablation_W_classification_h12_2023.joblib | `19e18a230262acc88cd0b71554b0e230ccb422444743b2c1695c9a57a6027ed7` | 5,958,168 |
| models/ablation/ablation_W_classification_h24_2023.joblib | `6097cb7ea83e4458bb4423ab905106dd8970d68f6f9137c7f953af7edf02a855` | 4,257,928 |
| models/ablation/ablation_W_classification_h48_2023.joblib | `7ff931e780d866f49cb8e850494780213d0427c993606eefd034c01038da7a0b` | 3,868,152 |
| models/ablation/ablation_W_classification_h6_2023.joblib | `e4bd8fe961f237b18da93b846c64e7592f8d1ad1b1bb3442b1973d05466b1fc1` | 7,624,888 |

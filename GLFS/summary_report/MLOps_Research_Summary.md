# MLOps Research Study

## Automated Research Summary

- Generated: 2026-08-23T11:37:27+03:30
- Dataset: `G:\Researching\GT_GL_MLops\GLFS_MLops_2026\data\raw\samples.csv`
- Dataset SHA256: `daf22e9b3887a8070242eec567c90206a066ba9990c40eaa4abc15b31d6b80e6`
- Active params: `G:\Researching\GT_GL_MLops\GLFS_MLops_2026\params.yaml`
- Targets: AC
- Grouping enabled: False
- Group column: Location_No
- Train/test: 0.8/0.2
- Classification enabled: True
- PDF status: generated successfully

## Analysis availability

| analysis | status | files_found | location |
| --- | --- | --- | --- |
| Data quality | Available | 5 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\data_processing_report\dq_report |
| Multicollinearity | Available | 4 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\data_processing_report\multicollinearity |
| Regression nested CV | Available | 48 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv |
| Regression learning curves | Available | 48 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves |
| Regression permutation | Available | 48 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity |
| Classification nested CV | Available | 36 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\nested_cv |
| Classification learning curves | Available | 36 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\learning_curves |
| Classification permutation | Available | 36 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\permutation_sensitivity |

## Selected / representative model results

| task | target | feature_set | model | selection_metric | cv_score | cv_sd | test_score | generalization_gap | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | XGB | R2 | 0.5002588938253897 | 0.2084147391506068 | 0.5600704207821465 | 0.4092332487378095 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs2 | DT | R2 | 0.6143072642188742 | 0.3249396187319774 | 0.6690719055920569 | 0.2612400153735662 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs3 | DT | R2 | 0.6096726017537303 | 0.1827677520778435 | 0.5001258544643864 | 0.1994422192135656 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs4 | XGB | R2 | 0.4889021437012776 | 0.1333576272122558 | 0.556405918608309 | 0.4337848352078943 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\all_evaluations.csv |
| classification | AC | fs1 | RF | Macro-F1 | 0.65996 | 0.167281 | 0.830686 | 0.321399 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs2 | DT | Macro-F1 | 0.718775 | 0.060897 | 0.843254 | 0.146811 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs3 | DT | Macro-F1 | 0.70959 | 0.103837 | 0.891659 | 0.097355 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs4 | XGB | Macro-F1 | 0.62086 | 0.168331 | 0.766667 | 0.30572 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report\all_evaluations_class.csv |

## Nested cross-validation

Nested CV summarizes saved outer-fold generalization estimates; hyperparameter optimization remains inside the inner CV loop.

| task | target | feature_set | model | folds | metric | mean_score | sd_score | secondary_metric | secondary_mean | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | ANN | 5 | R2 | -0.057817735798292046 | 0.7100262225862252 | RMSE | 0.40038014448886167 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\ANN_nested_cv_folds.csv |
| regression | AC | fs1 | DT | 5 | R2 | -0.23561956051317057 | 0.5388679799421482 | RMSE | 0.4238874037326168 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\DT_nested_cv_folds.csv |
| regression | AC | fs1 | EN | 5 | R2 | 0.06698982011385243 | 0.34723656090691435 | RMSE | 0.387844756938305 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\EN_nested_cv_folds.csv |
| regression | AC | fs1 | ET | 5 | R2 | 0.19661523191583608 | 0.37694322456997376 | RMSE | 0.353266391588232 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\ET_nested_cv_folds.csv |
| regression | AC | fs1 | GBR | 5 | R2 | 0.3411937181541648 | 0.3789294653812123 | RMSE | 0.3144124469561798 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\GBR_nested_cv_folds.csv |
| regression | AC | fs1 | KNN | 3 | R2 | 0.11741096078485325 | 0.2624713271023568 | RMSE | 0.37375951731255824 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\KNN_nested_cv_folds.csv |
| regression | AC | fs1 | LASSO | 5 | R2 | 0.0877430599087243 | 0.2274180193613014 | RMSE | 0.3861380053663378 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\LASSO_nested_cv_folds.csv |
| regression | AC | fs1 | MLR | 5 | R2 | 0.014821654825576059 | 0.5704414853891546 | RMSE | 0.37574646669503975 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\MLR_nested_cv_folds.csv |
| regression | AC | fs1 | RF | 5 | R2 | 0.4112491095448794 | 0.3480531490069077 | RMSE | 0.2966561663603828 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\RF_nested_cv_folds.csv |
| regression | AC | fs1 | RIDGE | 5 | R2 | 0.014095513207032844 | 0.4567208926313407 | RMSE | 0.3862498814461547 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\RIDGE_nested_cv_folds.csv |
| regression | AC | fs1 | SVR | 5 | R2 | 0.26290836363596937 | 0.35010704549334887 | RMSE | 0.34077619098194323 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\SVR_nested_cv_folds.csv |
| regression | AC | fs1 | XGB | 5 | R2 | 0.4687220318909248 | 0.24802608910606133 | RMSE | 0.2880383581729965 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs1\XGB_nested_cv_folds.csv |
| regression | AC | fs2 | ANN | 5 | R2 | 0.15537720428724797 | 0.5464915103297034 | RMSE | 0.34897408432747967 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\ANN_nested_cv_folds.csv |
| regression | AC | fs2 | DT | 5 | R2 | 0.3508555176836188 | 0.49665205296497295 | RMSE | 0.2900626277160427 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\DT_nested_cv_folds.csv |
| regression | AC | fs2 | EN | 5 | R2 | 0.2050437421405082 | 0.38869766125192207 | RMSE | 0.346266784647121 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\EN_nested_cv_folds.csv |
| regression | AC | fs2 | ET | 5 | R2 | 0.22937340925425267 | 0.4276303975434499 | RMSE | 0.3434119327130601 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\ET_nested_cv_folds.csv |
| regression | AC | fs2 | GBR | 5 | R2 | 0.5654023988088122 | 0.24921372628517732 | RMSE | 0.24559148407654732 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\GBR_nested_cv_folds.csv |
| regression | AC | fs2 | KNN | 3 | R2 | 0.014078758039105332 | 0.16223876329636738 | RMSE | 0.4017285179117285 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\KNN_nested_cv_folds.csv |
| regression | AC | fs2 | LASSO | 5 | R2 | 0.028823541081897875 | 0.3980890132591012 | RMSE | 0.3935673653273452 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\LASSO_nested_cv_folds.csv |
| regression | AC | fs2 | MLR | 5 | R2 | 0.11245159828683163 | 0.5473509058125805 | RMSE | 0.351385225939035 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\MLR_nested_cv_folds.csv |
| regression | AC | fs2 | RF | 5 | R2 | 0.5433279912734761 | 0.2230182630616579 | RMSE | 0.2611864376115715 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\RF_nested_cv_folds.csv |
| regression | AC | fs2 | RIDGE | 5 | R2 | 0.07910893784289832 | 0.4580084268331753 | RMSE | 0.3727422930437966 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\RIDGE_nested_cv_folds.csv |
| regression | AC | fs2 | SVR | 5 | R2 | 0.13239537164475385 | 0.4893940126198755 | RMSE | 0.3627269941178357 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\SVR_nested_cv_folds.csv |
| regression | AC | fs2 | XGB | 5 | R2 | 0.5363808762445321 | 0.210194135002093 | RMSE | 0.26548991671229255 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs2\XGB_nested_cv_folds.csv |
| regression | AC | fs3 | ANN | 5 | R2 | 0.10740780796020881 | 0.5253453105614435 | RMSE | 0.3609136472052003 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\ANN_nested_cv_folds.csv |
| regression | AC | fs3 | DT | 5 | R2 | 0.373579356138963 | 0.2080839471039912 | RMSE | 0.3189488366056924 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\DT_nested_cv_folds.csv |
| regression | AC | fs3 | EN | 5 | R2 | -0.08558839153321963 | 0.38935544252911797 | RMSE | 0.42232800456912506 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\EN_nested_cv_folds.csv |
| regression | AC | fs3 | ET | 5 | R2 | 0.22112631083180076 | 0.38274316367204814 | RMSE | 0.3481606363277959 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\ET_nested_cv_folds.csv |
| regression | AC | fs3 | GBR | 5 | R2 | 0.4578302219770502 | 0.3554935362830941 | RMSE | 0.2745394393705266 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\GBR_nested_cv_folds.csv |
| regression | AC | fs3 | KNN | 3 | R2 | 0.010537895007003667 | 0.14587528670770306 | RMSE | 0.4041462951129504 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\nested_cv\AC\fs3\KNN_nested_cv_folds.csv |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Learning-curve analysis

| task | target | feature_set | model | initial_train_size | final_train_size | initial_train_r2 | final_train_r2 | initial_cv_r2 | final_cv_r2 | cv_r2_gain | recent_cv_r2_gain | final_train_cv_gap | max_abs_train_cv_gap | initial_train_sd | final_train_sd | train_sd_change | initial_cv_sd | final_cv_sd | cv_sd_change | validation_trend | final_gap_level | more_data_likely_helpful | source_file | initial_train_macro_f1 | final_train_macro_f1 | initial_cv_macro_f1 | final_cv_macro_f1 | cv_macro_f1_gain | recent_cv_macro_f1_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | XGB | 34 | 206 | 0.971871947236682 | 0.9197884112873268 | -0.0620181501735528 | 0.5002588938253897 | 0.5622770439989425 | 0.0196558275707051 | 0.4195295174619371 | 1.0338900974102347 | 0.0135410965729261 | 0.0085955994958092 | -0.0049454970771169 | 0.4244647053772136 | 0.2084147391506068 | -0.2160499662266068 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | GBR | 34 | 206 | 0.7519692919928276 | 0.7812256506325262 | -0.0251643978314418 | 0.456643396415595 | 0.4818077942470368 | 0.0574726107246441 | 0.3245822542169312 | 0.7771336898242693 | 0.1298384784707646 | 0.0363002060607151 | -0.0935382724100495 | 0.5795691730360372 | 0.2674604935689365 | -0.3121086794671007 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | DT | 34 | 206 | 0.6579315360583067 | 0.8042208320588106 | -0.5425364605245757 | 0.4412194547271918 | 0.9837559152517676 | 0.1512802244569867 | 0.3630013773316187 | 1.2004679965828824 | 0.1777900503361602 | 0.0572865868169936 | -0.1205034635191665 | 1.281725982416436 | 0.1875060100891776 | -1.0942199723272583 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | RF | 34 | 206 | 0.7173018727189174 | 0.9020509196205484 | 0.1890517457213625 | 0.433329778603 | 0.2442780328816375 | -0.0054997309993503 | 0.4687211410175484 | 0.5282501269975549 | 0.150916179844239 | 0.0125842968808196 | -0.1383318829634194 | 0.3948917768868373 | 0.3712501093177667 | -0.0236416675690706 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | ET | 34 | 206 | 0.5273005989670219 | 0.7153050542347887 | 0.1402171233344512 | 0.2835602595369806 | 0.1433431362025294 | -0.0234335497629339 | 0.4317447946978081 | 0.4434612336647269 | 0.2204543853085351 | 0.0553307703250016 | -0.1651236149835334 | 0.3089852886548511 | 0.3563411065664745 | 0.0473558179116234 | improving | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | SVR | 34 | 206 | 0.3355320943268095 | 0.5028237208788292 | 0.1619589578273806 | 0.2795613400211941 | 0.1176023821938135 | -0.0166537459841432 | 0.2232623808576351 | 0.2232623808576351 | 0.2521024526972708 | 0.119292481074727 | -0.1328099716225437 | 0.1909326046475015 | 0.3168144826468441 | 0.1258818779993425 | improving | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | ANN | 34 | 206 | 0.4387743689965026 | 0.6473712886197942 | -1.5136441634768136 | 0.259044345893771 | 1.7726885093705846 | 0.0047174248598397 | 0.3883269427260232 | 1.9524185324733163 | 0.4027917194911987 | 0.1480205392279769 | -0.2547711802632217 | 1.9199331976846947 | 0.3849035364603317 | -1.535029661224363 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | KNN | 30 | 171 | 1.0 | 1.0 | -0.6487091084185448 | 0.1157780941594405 | 0.7644872025779853 | 0.0258643093057686 | 0.8842219058405595 | 1.6487091084185448 | 1.5700924586837752e-16 | 0.0 | -1.5700924586837752e-16 | 0.3805932388787045 | 0.2459104656739143 | -0.1346827732047902 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | EN | 34 | 206 | 0.4060681966679246 | 0.3828825787321205 | 0.0940651792029682 | 0.1043484827878676 | 0.0102833035848994 | -0.058827247707109 | 0.2785340959442529 | 0.4507783492284968 | 0.3287004178824163 | 0.1471563883820767 | -0.1815440295003395 | 0.3421044362079724 | 0.3956656169377048 | 0.0535611807297324 | near plateau | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | LASSO | 34 | 206 | 0.3049129310847394 | 0.3258832858277148 | 0.0678282837070542 | 0.0873577630409619 | 0.0195294793339077 | -0.085704376745975 | 0.2385255227867529 | 0.3083972136472921 | 0.3297243502208372 | 0.1558451158435246 | -0.1738792343773126 | 0.3016416551039334 | 0.2277660918437258 | -0.0738755632602076 | near plateau | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | RIDGE | 34 | 206 | 0.2803063877671514 | 0.377604409178213 | 0.0644602759943069 | 0.0512614060999592 | -0.0131988698943476 | -0.028367597466541 | 0.3263430030782537 | 0.3698070087758181 | 0.2276676314977097 | 0.1329025966714091 | -0.0947650348263005 | 0.22100133292643 | 0.4345019145708753 | 0.2135005816444452 | near plateau | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | MLR | 34 | 206 | 0.4759418272525119 | 0.4298986618094856 | 0.1549359207324923 | 0.014821654825576 | -0.1401142659069162 | 0.008314651252163 | 0.4150770069839096 | 0.6804983256639758 | 0.3217497325745261 | 0.1354692455894973 | -0.1862804869850288 | 0.4271934760153512 | 0.5704414853891546 | 0.1432480093738034 | declining | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | DT | 34 | 206 | 0.5733693874177592 | 0.8619554414022259 | -0.1175452971182854 | 0.6143072642188742 | 0.7318525613371596 | 0.3078926778986238 | 0.2476481771833517 | 0.6909146845360447 | 0.173959308864937 | 0.0350788869738947 | -0.1388804218910423 | 0.3433565622122538 | 0.3249396187319774 | -0.0184169434802763 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | XGB | 34 | 206 | 0.9841129107013092 | 0.9849960868372908 | -0.1275175867831761 | 0.6082511519782425 | 0.7357687387614187 | 0.0415323945500056 | 0.3767449348590482 | 1.1116304974844853 | 0.0088439268319132 | 0.0038487800416599 | -0.0049951467902533 | 0.4377491320980082 | 0.238230510712516 | -0.1995186213854922 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | GBR | 34 | 206 | 0.9027294844099008 | 0.9223035691734432 | 0.106801044775165 | 0.5811046877481046 | 0.4743036429729396 | 0.067615416830402 | 0.3411988814253386 | 0.7959284396347357 | 0.0525954480365033 | 0.0163797314626238 | -0.0362157165738795 | 0.4023800039323149 | 0.2080845686737585 | -0.1942954352585564 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | RF | 34 | 206 | 0.6742194165910809 | 0.8886859936121263 | 0.1872201709638441 | 0.5536282982700518 | 0.3664081273062076 | 0.0874542499886184 | 0.3350576953420745 | 0.4869992456272367 | 0.1583759937225674 | 0.0238807953054279 | -0.1344951984171394 | 0.2881875161709156 | 0.2345423812531641 | -0.0536451349177515 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | ET | 34 | 206 | 0.3116222497166639 | 0.7109657077996137 | 0.1145256090072881 | 0.2739192500261213 | 0.1593936410188332 | -0.0430066070398327 | 0.4370464577734924 | 0.4370464577734924 | 0.253186557836342 | 0.052996559317534 | -0.2001899985188079 | 0.1823870869803277 | 0.2907906384718735 | 0.1084035514915457 | improving | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | ANN | 34 | 206 | -0.0312960121459394 | 0.403667610767375 | -0.4437598348501524 | 0.272263155644769 | 0.7160229904949214 | 0.068904640293145 | 0.131404455122606 | 0.5301964973224974 | 1.0259903519727682 | 0.2140022828570486 | -0.8119880691157196 | 1.3222350676977592 | 0.4360230597384379 | -0.8862120079593213 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | SVR | 34 | 206 | 0.349845484804644 | 0.5443265595496547 | 0.1151628346487545 | 0.2700789503072071 | 0.1549161156584526 | -0.0094525596623994 | 0.2742476092424476 | 0.2742476092424476 | 0.2517301316302004 | 0.1000138137990893 | -0.1517163178311111 | 0.1832418618914182 | 0.3155218703109867 | 0.1322800084195685 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | EN | 34 | 206 | 0.4617671795483424 | 0.4498599944267816 | 0.170756478478137 | 0.1723810667031242 | 0.0016245882249872 | -0.0376178748826477 | 0.2774789277236574 | 0.4294334646399414 | 0.3047759219680578 | 0.1393922744582749 | -0.1653836475097829 | 0.4348306053157756 | 0.4258206937638599 | -0.0090099115519157 | near plateau | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | LASSO | 34 | 206 | 0.4968984189706025 | 0.4767264286039882 | 0.1976246891988319 | 0.1570950584179745 | -0.0405296307808573 | -0.0257868060268476 | 0.3196313701860137 | 0.475092730620925 | 0.304520487469461 | 0.131114270900205 | -0.1734062165692559 | 0.4916493631066624 | 0.5117921046365054 | 0.0201427415298429 | declining | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | RIDGE | 34 | 206 | 0.2948624656875939 | 0.4278109173960736 | 0.0792810107025977 | 0.1313892844101477 | 0.0521082737075499 | -0.0334001958227428 | 0.2964216329859259 | 0.3607449747049416 | 0.2321354096737916 | 0.1307924045580307 | -0.1013430051157608 | 0.2417949764064196 | 0.4052377445822085 | 0.1634427681757889 | improving | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | MLR | 34 | 206 | 0.5057606670843138 | 0.4806966418198012 | 0.1975022454384501 | 0.1124515982868316 | -0.0850506471516185 | -0.0269634871022416 | 0.3682450435329696 | 0.5223063834774833 | 0.3030556742661847 | 0.1289789454478939 | -0.1740767288182907 | 0.5052661599503665 | 0.5473509058125805 | 0.042084745862214 | declining | large | False | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | KNN | 30 | 171 | 1.0 | 1.0 | -0.5610718198888928 | 0.0393855404702749 | 0.6004573603591676 | 0.0115590774656238 | 0.9606144595297252 | 1.5610718198888929 | 0.0 | 0.0 | 0.0 | 0.2742754087385961 | 0.1774315976292456 | -0.0968438111093505 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | DT | 34 | 206 | 0.6849626106234754 | 0.8260805177037078 | -0.0376853596440962 | 0.6096726017537303 | 0.6473579613978265 | 0.4115201019840416 | 0.2164079159499775 | 0.8634495943931676 | 0.1843953932692134 | 0.0567182462777092 | -0.1276771469915041 | 0.5275768006830613 | 0.1827677520778435 | -0.3448090486052179 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | XGB | 34 | 206 | 0.9854874093660412 | 0.985976363223138 | 0.146785262579411 | 0.5798696082550886 | 0.4330843456756776 | 0.1080864516538426 | 0.4061067549680494 | 0.8578061981114854 | 0.0091306082201199 | 0.0039494900055552 | -0.0051811182145646 | 0.3443209744642207 | 0.2447342796283065 | -0.0995866948359142 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | GBR | 34 | 206 | 0.770507646558497 | 0.8685302990354031 | 0.2209011365611172 | 0.5676845186751759 | 0.3467833821140587 | 0.1222011737904794 | 0.3008457803602272 | 0.5496065099973798 | 0.1196343887256315 | 0.0270750048825523 | -0.0925593838430792 | 0.3163495629907443 | 0.1873181048750571 | -0.1290314581156872 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | RF | 34 | 206 | 0.6520730795417711 | 0.8975969906392839 | 0.1652821785279178 | 0.5324271633391072 | 0.3671449848111894 | 0.1263665624072522 | 0.3651698273001766 | 0.5011725967345741 | 0.1548986308121128 | 0.0200923156118882 | -0.1348063152002246 | 0.2861759941160143 | 0.2696286422481782 | -0.0165473518678361 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | ANN | 34 | 206 | -1.384919868886422 | 0.7379137550474255 | -1.4347556899733749 | 0.3180692122883425 | 1.7528249022617173 | 0.1320473345923222 | 0.419844542759083 | 0.5401819572742402 | 4.662741885467874 | 0.0513891524462223 | -4.611352733021652 | 2.132359056552301 | 0.3440866944141968 | -1.7882723621381045 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | SVR | 34 | 206 | 0.3902887829147856 | 0.5842732112156879 | 0.0708586257031669 | 0.2789798745424994 | 0.2081212488393325 | 0.0017189436100361 | 0.3052933366731885 | 0.3760331411244388 | 0.2099634981583027 | 0.0827443785836246 | -0.1272191195746781 | 0.1567181984372236 | 0.2533031601017073 | 0.0965849616644836 | improving | large | True | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Permutation sensitivity / feature importance

Permutation results quantify predictive dependence on each feature and are not causal effects.

| task | target | feature_set | model | rank | feature | mean_importance | relative_contribution_percent | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | ANN | 1 | PI/FF | 0.4457652816949636 | 39.87723629086136 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | ANN | 2 | Distance | 0.4196452011698614 | 37.54058813586083 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | ANN | 3 | Depth | 0.1777696305933908 | 15.902901943275936 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | ANN | 4 | Ucs_class | 0.0746638575822731 | 6.679273630001871 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 1 | Depth | 1.1224865692339323 | 75.17744704404944 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 2 | Distance | 0.2803698227281194 | 18.777496389357747 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 3 | PI/FF | 0.0600686636384283 | 4.023040367214982 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 4 | Ucs_class | 0.0301910495210773 | 2.022016199377839 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 1 | PI/FF | 0.5584163610754089 | 93.44829110847697 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 2 | Depth | 0.039150865089472 | 6.551708891523028 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 3 | Distance | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 4 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 1 | PI/FF | 0.4777792810378648 | 43.70759211051328 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 2 | Distance | 0.3264700050136985 | 29.86571076179652 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 3 | Depth | 0.2151953869278761 | 19.686228702668 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 4 | Ucs_class | 0.0736818479915963 | 6.740468425022196 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 1 | Depth | 0.476488805701711 | 49.81344273987497 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 2 | PI/FF | 0.3921488697154431 | 40.99631519844281 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 3 | Distance | 0.0879089503399295 | 9.190242061682223 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 4 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 1 | PI/FF | 1.0209803995967106 | 39.79755853554344 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 2 | Distance | 0.7072967953780895 | 27.57025073858479 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 3 | Depth | 0.5468747511956468 | 21.317039906854667 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 4 | Ucs_class | 0.2902828120569122 | 11.315150819017092 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 1 | PI/FF | 0.4837026185884371 | 100.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 2 | Distance | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 3 | Depth | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 4 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | MLR | 1 | PI/FF | 0.9181851616889464 | 90.00972520598984 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\MLR_permutation_sensitivity.csv |
| regression | AC | fs1 | MLR | 2 | Depth | 0.066628048105002 | 6.531550008836201 | G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\MLR_permutation_sensitivity.csv |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Reproducibility

- Python: `3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.19045-SP0`
- Regression reports: `G:\Researching\GT_GL_MLops\GLFS_MLops_2026\regression_report`
- Classification reports: `G:\Researching\GT_GL_MLops\GLFS_MLops_2026\classification_report`

### Package versions

- numpy: 2.5.2
- pandas: 2.3.3
- scipy: 1.18.0
- scikit-learn: 1.9.0
- xgboost: 3.4.1
- statsmodels: 0.14.6
- matplotlib: 3.11.1
- joblib: 1.5.3
- PyYAML: 6.0.3
- reportlab: 5.0.1

## Interpretation note

The independent test partition is final holdout evidence and must not be used as a tuning criterion. Nested CV, learning curves, and permutation sensitivity answer different questions and should not be collapsed into one score.

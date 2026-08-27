# MLOps Research Study

## Automated Research Summary

- Generated: 2026-08-23T10:59:22+03:30
- Dataset: `G:\Researching\GT_GL_MLops\GTFS_MLops_2026\data\raw\samples.csv`
- Dataset SHA256: `daf22e9b3887a8070242eec567c90206a066ba9990c40eaa4abc15b31d6b80e6`
- Active params: `G:\Researching\GT_GL_MLops\GTFS_MLops_2026\params.yaml`
- Targets: AC
- Grouping enabled: False
- Group column: Location_No
- Train/test: 0.8/0.2
- Classification enabled: True
- PDF status: generated successfully

## Analysis availability

| analysis | status | files_found | location |
| --- | --- | --- | --- |
| Data quality | Available | 5 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\data_processing_report\dq_report |
| Multicollinearity | Available | 4 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\data_processing_report\multicollinearity |
| Regression nested CV | Available | 48 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv |
| Regression learning curves | Available | 46 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves |
| Regression permutation | Available | 48 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity |
| Classification nested CV | Available | 36 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\nested_cv |
| Classification learning curves | Available | 36 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\learning_curves |
| Classification permutation | Available | 36 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\permutation_sensitivity |

## Selected / representative model results

| task | target | feature_set | model | selection_metric | cv_score | cv_sd | test_score | generalization_gap | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | SVR | R2 | 0.2601581937583264 | 0.2592532079813409 | 0.3579777903550181 | 0.1501439671327215 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs2 | SVR | R2 | 0.2789994379707864 | 0.2416810501780325 | 0.340591303591592 | 0.1577258750465674 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs3 | SVR | R2 | 0.2835503075477345 | 0.2553727040984295 | 0.3184473899931648 | 0.1730307934379456 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\all_evaluations.csv |
| regression | AC | fs4 | SVR | R2 | 0.0959826395092362 | 0.1664084036739288 | 0.1858566490005502 | 0.1225468633264936 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\all_evaluations.csv |
| classification | AC | fs1 | XGB | Macro-F1 | 0.649244 | 0.104533 | 0.78394 | 0.068157 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs2 | SVC | Macro-F1 | 0.618984 | 0.149788 | 0.674835 | 0.039656 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs3 | SVC | Macro-F1 | 0.57509 | 0.156192 | 0.643782 | 0.10113 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\all_evaluations_class.csv |
| classification | AC | fs4 | ANN | Macro-F1 | 0.499649 | 0.10344 | 0.562963 | 0.06333 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report\all_evaluations_class.csv |

## Nested cross-validation

Nested CV summarizes saved outer-fold generalization estimates; hyperparameter optimization remains inside the inner CV loop.

| task | target | feature_set | model | folds | metric | mean_score | sd_score | secondary_metric | secondary_mean | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | ANN | 5 | R2 | 0.18824020709038453 | 0.3143254560992334 | RMSE | 0.35161433480729565 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\ANN_nested_cv_folds.csv |
| regression | AC | fs1 | DT | 5 | R2 | 0.02603853989985891 | 0.3635682130907393 | RMSE | 0.3863708487876332 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\DT_nested_cv_folds.csv |
| regression | AC | fs1 | EN | 5 | R2 | 0.08364071887909937 | 0.4430820508501307 | RMSE | 0.3666851723307486 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\EN_nested_cv_folds.csv |
| regression | AC | fs1 | ET | 5 | R2 | 0.1194538281140479 | 0.29508567778646816 | RMSE | 0.36853485209753656 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\ET_nested_cv_folds.csv |
| regression | AC | fs1 | GBR | 5 | R2 | 0.17109640971235635 | 0.30028328977054086 | RMSE | 0.36405345422627305 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\GBR_nested_cv_folds.csv |
| regression | AC | fs1 | KNN | 3 | R2 | -0.09000410759528772 | 0.10356449014161967 | RMSE | 0.41084975665939716 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\KNN_nested_cv_folds.csv |
| regression | AC | fs1 | LASSO | 5 | R2 | 0.09044346290410857 | 0.4494533610799976 | RMSE | 0.36471061187869125 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\LASSO_nested_cv_folds.csv |
| regression | AC | fs1 | MLR | 5 | R2 | 0.09745817362343759 | 0.461608394191044 | RMSE | 0.36222631299539376 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\MLR_nested_cv_folds.csv |
| regression | AC | fs1 | RF | 5 | R2 | 0.033170216668857846 | 0.36865514853709397 | RMSE | 0.3779760657109847 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\RF_nested_cv_folds.csv |
| regression | AC | fs1 | RIDGE | 5 | R2 | 0.09159457635829431 | 0.4545044210005521 | RMSE | 0.3641082344207881 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\RIDGE_nested_cv_folds.csv |
| regression | AC | fs1 | SVR | 5 | R2 | 0.1477239980086784 | 0.3472970405509121 | RMSE | 0.36443888122697743 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\SVR_nested_cv_folds.csv |
| regression | AC | fs1 | XGB | 5 | R2 | 0.1093468269833836 | 0.24170034808835172 | RMSE | 0.37390922369455604 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs1\XGB_nested_cv_folds.csv |
| regression | AC | fs2 | ANN | 5 | R2 | -0.01705724064154221 | 0.41019968810902185 | RMSE | 0.3884711337815472 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\ANN_nested_cv_folds.csv |
| regression | AC | fs2 | DT | 5 | R2 | -0.041441973213466585 | 0.4769629013862566 | RMSE | 0.3900637229843664 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\DT_nested_cv_folds.csv |
| regression | AC | fs2 | EN | 5 | R2 | 0.24412138259324773 | 0.3809655839509368 | RMSE | 0.33811194851227255 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\EN_nested_cv_folds.csv |
| regression | AC | fs2 | ET | 5 | R2 | 0.1283762092223267 | 0.1541105409701501 | RMSE | 0.3774227292740965 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\ET_nested_cv_folds.csv |
| regression | AC | fs2 | GBR | 5 | R2 | 0.12039446484436497 | 0.23516571670267616 | RMSE | 0.3766006110736019 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\GBR_nested_cv_folds.csv |
| regression | AC | fs2 | KNN | 3 | R2 | -0.11240577858649764 | 0.1577000490565461 | RMSE | 0.4248251645313171 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\KNN_nested_cv_folds.csv |
| regression | AC | fs2 | LASSO | 5 | R2 | 0.2591736711462248 | 0.38701753227130087 | RMSE | 0.3334357782910019 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\LASSO_nested_cv_folds.csv |
| regression | AC | fs2 | MLR | 5 | R2 | 0.24804078273728547 | 0.41265861411430504 | RMSE | 0.3342979410862036 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\MLR_nested_cv_folds.csv |
| regression | AC | fs2 | RF | 5 | R2 | 0.10576311318755771 | 0.2089715735864512 | RMSE | 0.37640035863378535 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\RF_nested_cv_folds.csv |
| regression | AC | fs2 | RIDGE | 5 | R2 | 0.21337996833297712 | 0.3762359363844414 | RMSE | 0.3463296341608049 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\RIDGE_nested_cv_folds.csv |
| regression | AC | fs2 | SVR | 5 | R2 | 0.08012416814717395 | 0.3693049543279168 | RMSE | 0.383417752168708 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\SVR_nested_cv_folds.csv |
| regression | AC | fs2 | XGB | 5 | R2 | -0.04575506631708266 | 0.11071142196516166 | RMSE | 0.41172231507587176 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs2\XGB_nested_cv_folds.csv |
| regression | AC | fs3 | ANN | 5 | R2 | -0.09701419897696155 | 0.4673102496105836 | RMSE | 0.42578894344423 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\ANN_nested_cv_folds.csv |
| regression | AC | fs3 | DT | 5 | R2 | -0.23987634546630723 | 0.3023347301023863 | RMSE | 0.4569430141347497 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\DT_nested_cv_folds.csv |
| regression | AC | fs3 | EN | 5 | R2 | 0.047349939832706966 | 0.500773571188615 | RMSE | 0.3841285312524003 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\EN_nested_cv_folds.csv |
| regression | AC | fs3 | ET | 5 | R2 | 0.10329376228881955 | 0.16822354810895657 | RMSE | 0.383236865214279 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\ET_nested_cv_folds.csv |
| regression | AC | fs3 | GBR | 5 | R2 | -0.014443520139493548 | 0.21980132525849544 | RMSE | 0.4086506557031223 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\GBR_nested_cv_folds.csv |
| regression | AC | fs3 | KNN | 3 | R2 | -0.15493358084908596 | 0.1456710165837343 | RMSE | 0.42573562499491896 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\nested_cv\AC\fs3\KNN_nested_cv_folds.csv |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Learning-curve analysis

| task | target | feature_set | model | initial_train_size | final_train_size | initial_train_r2 | final_train_r2 | initial_cv_r2 | final_cv_r2 | cv_r2_gain | recent_cv_r2_gain | final_train_cv_gap | max_abs_train_cv_gap | initial_train_sd | final_train_sd | train_sd_change | initial_cv_sd | final_cv_sd | cv_sd_change | validation_trend | final_gap_level | more_data_likely_helpful | source_file | initial_train_macro_f1 | final_train_macro_f1 | initial_cv_macro_f1 | final_cv_macro_f1 | cv_macro_f1_gain | recent_cv_macro_f1_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | SVR | 41 | 206 | 0.30547709799939 | 0.4019211198598807 | 0.1173932593619145 | 0.2601581937583264 | 0.1427649343964118 | -0.0061223076333858 | 0.1417629261015543 | 0.1880838386374755 | 0.2041751410637679 | 0.142744890626924 | -0.0614302504368439 | 0.1225902070673509 | 0.2592532079813409 | 0.1366630009139899 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | GBR | 41 | 206 | 0.5088538658140627 | 0.4435016377247896 | -0.2973481748233099 | 0.2282197976726208 | 0.5255679724959307 | -0.0599898236253835 | 0.2152818400521688 | 0.8062020406373726 | 0.1500707683792716 | 0.1500148783289474 | -5.58900503242199e-05 | 0.9734310733459768 | 0.3181445600219228 | -0.655286513324054 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | XGB | 41 | 206 | 0.284086278389476 | 0.4661669347522603 | -0.3781782911967706 | 0.1693025106388234 | 0.547480801835594 | -0.0357010161455627 | 0.2968644241134369 | 0.6622645695862466 | 0.1779235181637815 | 0.1377304110478608 | -0.0401931071159206 | 0.8421425347839058 | 0.2467333974648602 | -0.5954091373190455 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | DT | 41 | 206 | 0.3442701131401987 | 0.4089171275646072 | -0.5482814306828594 | 0.1469628983098785 | 0.6952443289927379 | 0.0500159299269439 | 0.2619542292547286 | 0.892551543823058 | 0.1508223712878789 | 0.142488797441751 | -0.0083335738461279 | 1.2563443981369964 | 0.4484988935352933 | -0.807845504601703 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | ET | 41 | 206 | 0.4324104507334602 | 0.4113667638365652 | -0.377994032339736 | 0.1416273602266291 | 0.5196213925663651 | 0.0092718759908664 | 0.2697394036099361 | 0.8104044830731962 | 0.1596202323471204 | 0.1351661012072854 | -0.0244541311398349 | 0.775773458775398 | 0.1828484625629956 | -0.5929249962124025 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | EN | 41 | 206 | 0.4560805319486188 | 0.3669525032378787 | -0.099069029255279 | 0.1337515089900139 | 0.2328205382452929 | -0.0735237066797381 | 0.2332009942478648 | 0.5551495612038978 | 0.324575759949276 | 0.1518282418472403 | -0.1727475181020357 | 0.5755155445323008 | 0.3084365472089246 | -0.2670789973233762 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | ANN | 41 | 206 | 0.1371100369675271 | 0.3754833579112311 | -0.5534931114439459 | 0.1286165640572006 | 0.6821096755011464 | 0.0050140108831043 | 0.2468667938540305 | 0.690603148411473 | 0.7557438813396006 | 0.1407307391994501 | -0.6150131421401505 | 0.6978782260239201 | 0.2986685780566677 | -0.3992096479672524 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | RIDGE | 41 | 206 | 0.2554759798814011 | 0.3338277496816614 | -0.1523395967108845 | 0.1167459344689588 | 0.2690855311798433 | 0.0089492012879539 | 0.2170818152127026 | 0.4078155765922856 | 0.1834855968312584 | 0.1338027875114072 | -0.0496828093198511 | 0.4065725435244384 | 0.2102834918805644 | -0.196289051643874 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | LASSO | 41 | 206 | 0.4714378908274702 | 0.3845453889226196 | -0.1463139587570553 | 0.1079414801082977 | 0.254255438865353 | -0.1148521107285115 | 0.2766039088143219 | 0.6177518495845254 | 0.3338340538283219 | 0.1509964222906654 | -0.1828376315376564 | 0.665068358044016 | 0.4167394990000934 | -0.2483288590439226 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | MLR | 41 | 206 | 0.4743533042738754 | 0.3868139302213912 | -0.1627966863322558 | 0.0974581736234375 | 0.2602548599556933 | -0.1173390083469582 | 0.2893557565979537 | 0.6371499906061312 | 0.334701905234915 | 0.1508845980409311 | -0.1838173071939839 | 0.6830986524543846 | 0.4616083941910439 | -0.2214902582633407 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs1 | RF | 41 | 206 | 0.5211143804350504 | 0.4958519992043203 | -0.463437667346924 | 0.0685262893148908 | 0.5319639566618148 | -0.1217541570697055 | 0.4273257098894295 | 0.9845520477819744 | 0.1376359563787871 | 0.1338793529151319 | -0.0037566034636551 | 1.229973598565702 | 0.3671195660629254 | -0.8628540325027767 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs1\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | SVR | 41 | 206 | 0.3287195590752472 | 0.4300806993161368 | 0.0435231990155574 | 0.2789994379707864 | 0.235476238955229 | 0.0418675936303442 | 0.1510812613453503 | 0.2851963600596898 | 0.1322272172947938 | 0.1376852277254556 | 0.0054580104306617 | 0.0748062906588654 | 0.2416810501780325 | 0.1668747595191671 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | ANN | 41 | 206 | 0.4478724073599586 | 0.4582157676378282 | -0.4053828758133582 | 0.2608646311782065 | 0.6662475069915648 | -0.0281485107529398 | 0.1973511364596216 | 0.8532552831733168 | 0.2340610152921575 | 0.1206513193197214 | -0.1134096959724361 | 0.5959502394012104 | 0.2965720690039895 | -0.2993781703972208 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | EN | 41 | 206 | 0.4861094567379609 | 0.4361088847467638 | -0.0510315885029989 | 0.2604424416283598 | 0.3114740301313586 | -0.0715868346111197 | 0.175666443118404 | 0.5371410452409598 | 0.313898044178535 | 0.1400565370184499 | -0.1738415071600851 | 0.478667563151441 | 0.3798281870719628 | -0.0988393760794781 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | LASSO | 41 | 206 | 0.4855595931853397 | 0.435936367622381 | -0.0516848691141609 | 0.2602931490451012 | 0.311978018159262 | -0.0707637262823416 | 0.1756432185772798 | 0.5372444622995006 | 0.313566170248553 | 0.1400791204484896 | -0.1734870498000633 | 0.4757060835184252 | 0.3769380821358545 | -0.0987680013825706 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | RIDGE | 41 | 206 | 0.4606536553122792 | 0.4364365727006831 | -0.0718602036787574 | 0.2564822863563787 | 0.3283424900351361 | -0.0703114801382198 | 0.1799542863443043 | 0.5325138589910365 | 0.2936833051284043 | 0.1397011167113736 | -0.1539821884170306 | 0.4401223561221563 | 0.3804815255109546 | -0.0596408306112016 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | MLR | 41 | 206 | 0.4903706331226655 | 0.437507239390335 | -0.0375513161865256 | 0.2480407827372854 | 0.285592098923811 | -0.0830825401241955 | 0.1894664566530496 | 0.5279219493091911 | 0.3162640495309449 | 0.1399850645313007 | -0.1762789849996442 | 0.4964467011008059 | 0.412658614114305 | -0.0837880869865009 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | GBR | 41 | 206 | 0.6337570840588567 | 0.4762810740753148 | -0.4008294148334408 | 0.2324548838403223 | 0.6332842986737631 | 0.0323742532389863 | 0.2438261902349924 | 1.0345864988922977 | 0.1412810137437298 | 0.1109291423205742 | -0.0303518714231556 | 0.8583049909315205 | 0.2455193655259858 | -0.6127856254055346 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | ET | 41 | 206 | 0.4429538326352396 | 0.459394971015894 | -0.2009722994967371 | 0.1826928515094625 | 0.3836651510061996 | 0.0643482394888636 | 0.2767021195064315 | 0.6439261321319767 | 0.1013271606223106 | 0.0875319049549462 | -0.0137952556673644 | 0.385190586184327 | 0.1587288186622819 | -0.2264617675220451 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | XGB | 41 | 206 | 0.6120217242058973 | 0.5630365153030319 | -0.3548671492036144 | 0.163619361677096 | 0.5184865108807104 | 0.0187503238561899 | 0.3994171536259359 | 0.9668888734095116 | 0.0421493200635316 | 0.0806850551481992 | 0.0385357350846675 | 0.6248092021386245 | 0.2203277650205892 | -0.4044814371180352 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | RF | 41 | 206 | 0.6047837287792259 | 0.5488387739196499 | -0.3849810565939688 | 0.1331188173331166 | 0.5180998739270855 | -0.0365732974955059 | 0.4157199565865332 | 0.9897647853731948 | 0.0777852784336799 | 0.0914483447418527 | 0.0136630663081728 | 0.764775391777641 | 0.2003112294930953 | -0.5644641622845457 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | DT | 41 | 206 | 0.6714226192903904 | 0.5106166543830147 | -0.6306920819664978 | 0.0316868318262984 | 0.6623789137927962 | -0.0339165021259111 | 0.4789298225567163 | 1.302114701256888 | 0.105646461691018 | 0.0786297959531756 | -0.0270166657378424 | 1.2287478658743334 | 0.1659460597401231 | -1.0628018061342104 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs2 | KNN | 28 | 171 | 0.6644388316801301 | 0.6416155835193619 | -0.2772584038506916 | -0.0320977712024318 | 0.2451606326482598 | 0.0302417109350374 | 0.6737133547217937 | 0.9416972355308216 | 0.2777485278107065 | 0.0974003553665301 | -0.1803481724441764 | 0.0840334619574118 | 0.1449492860856688 | 0.0609158241282569 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs2\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | SVR | 41 | 206 | 0.3480534255755694 | 0.4570080440360003 | 0.0465928317476101 | 0.2835503075477345 | 0.2369574758001244 | 0.0661052116309169 | 0.1734577364882658 | 0.3014605938279593 | 0.1027686034764367 | 0.1257505721925313 | 0.0229819687160946 | 0.1203851352137949 | 0.2553727040984295 | 0.1349875688846346 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | ANN | 41 | 206 | 0.5151877634318679 | 0.4447855800955447 | 0.0031934993184861 | 0.1946803355325697 | 0.1914868362140836 | 0.1227988638557163 | 0.250105244562975 | 0.5119942641133818 | 0.2424778056330881 | 0.2145454688075941 | -0.027932336825494 | 0.3520113015894275 | 0.3491106825975635 | -0.002900618991864 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | RIDGE | 41 | 206 | 0.2583376569425585 | 0.3860172769130601 | -0.0750165967650453 | 0.1851077240232643 | 0.2601243207883096 | 0.0501994872199847 | 0.2009095528897957 | 0.3333542537076038 | 0.1426900896211818 | 0.1299247801313336 | -0.0127653094898481 | 0.150144231325732 | 0.2305845015796651 | 0.080440270253933 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | ET | 41 | 206 | 0.7324851681864599 | 0.6000513631068778 | -0.1156346439872185 | 0.1801578589340296 | 0.2957925029212481 | 0.0610632950641275 | 0.4198935041728482 | 0.8481198121736784 | 0.0548510658674022 | 0.0851586455509075 | 0.0303075796835053 | 0.2131437748111336 | 0.2143302598427535 | 0.0011864850316198 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | GBR | 41 | 206 | 0.6334769564796632 | 0.4477989087304266 | -0.1705548298477311 | 0.1646450998306973 | 0.3351999296784284 | 0.0424072033631535 | 0.2831538088997293 | 0.8040317863273942 | 0.1460488350165431 | 0.0994645272502232 | -0.0465843077663198 | 0.2666459734729821 | 0.2333029128203075 | -0.0333430606526745 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | EN | 41 | 206 | 0.5252845765550029 | 0.4340478777051292 | 0.0699167606134098 | 0.1570004630119709 | 0.0870837023985611 | -0.043232932138931 | 0.2770474146931583 | 0.4553678159415931 | 0.261796718106293 | 0.149666504463946 | -0.112130213642347 | 0.4014148452841715 | 0.4029987764705375 | 0.001583931186366 | improving | large | False | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |
| regression | AC | fs3 | XGB | 41 | 206 | 0.6396282162574446 | 0.631982959382361 | -0.2089347944550919 | 0.1469161295494724 | 0.3558509240045643 | 0.0793851530535875 | 0.4850668298328886 | 0.8485630107125365 | 0.0717731162410786 | 0.0526914213738006 | -0.019081694867278 | 0.2936263250791939 | 0.156308958420673 | -0.1373173666585209 | improving | large | True | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\learning_curves\AC\fs3\learning_curve_model_summary.csv | None | None | None | None | None | None |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Permutation sensitivity / feature importance

Permutation results quantify predictive dependence on each feature and are not causal effects.

| task | target | feature_set | model | rank | feature | mean_importance | relative_contribution_percent | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | AC | fs1 | ANN | 1 | PI/FF | 0.6097172627409633 | 99.22578513091344 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | ANN | 2 | Ucs_class | -0.0047573538483973 | 0.774214869086542 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ANN_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 1 | PI/FF | 0.8791990573211108 | 100.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | DT | 2 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\DT_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 1 | PI/FF | 0.6489089038887704 | 100.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | EN | 2 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\EN_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 1 | PI/FF | 0.6359746392445589 | 99.36379954940962 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | ET | 2 | Ucs_class | 0.0040719794722648 | 0.636200450590374 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\ET_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 1 | PI/FF | 0.8619629822416394 | 100.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | GBR | 2 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\GBR_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 1 | PI/FF | 0.9400014991072064 | 96.83089215047804 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | KNN | 2 | Ucs_class | 0.0307646254539682 | 3.1691078495219567 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\KNN_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 1 | PI/FF | 0.8568354881921562 | 98.04571034217786 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | LASSO | 2 | Ucs_class | 0.0170788168822982 | 1.9542896578221356 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\LASSO_permutation_sensitivity.csv |
| regression | AC | fs1 | MLR | 1 | PI/FF | 0.9593337000421028 | 96.55950513189862 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\MLR_permutation_sensitivity.csv |
| regression | AC | fs1 | MLR | 2 | Ucs_class | 0.0341818515668967 | 3.4404948681013847 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\MLR_permutation_sensitivity.csv |
| regression | AC | fs1 | RF | 1 | PI/FF | 0.9828392987676297 | 97.6409765322551 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\RF_permutation_sensitivity.csv |
| regression | AC | fs1 | RF | 2 | Ucs_class | 0.0237455733561704 | 2.3590234677448967 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\RF_permutation_sensitivity.csv |
| regression | AC | fs1 | RIDGE | 1 | PI/FF | 0.5562653840205145 | 99.99292152041411 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\RIDGE_permutation_sensitivity.csv |
| regression | AC | fs1 | RIDGE | 2 | Ucs_class | -3.9377919009083344e-05 | 0.0070784795858648 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\RIDGE_permutation_sensitivity.csv |
| regression | AC | fs1 | SVR | 1 | PI/FF | 0.7332597115040609 | 99.74821521736952 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\SVR_permutation_sensitivity.csv |
| regression | AC | fs1 | SVR | 2 | Ucs_class | 0.0018508966468264 | 0.2517847826304923 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\SVR_permutation_sensitivity.csv |
| regression | AC | fs1 | XGB | 1 | PI/FF | 0.8179783237825264 | 100.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\XGB_permutation_sensitivity.csv |
| regression | AC | fs1 | XGB | 2 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs1\XGB_permutation_sensitivity.csv |
| regression | AC | fs2 | ANN | 1 | FF | 0.508043224003062 | 57.2397004028765 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\ANN_permutation_sensitivity.csv |
| regression | AC | fs2 | ANN | 2 | PI | 0.3371835385672606 | 37.98945407892189 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\ANN_permutation_sensitivity.csv |
| regression | AC | fs2 | ANN | 3 | Ucs_class | 0.0423446615064026 | 4.770845518201612 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\ANN_permutation_sensitivity.csv |
| regression | AC | fs2 | DT | 1 | FF | 0.6120409161263402 | 66.17606056791244 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\DT_permutation_sensitivity.csv |
| regression | AC | fs2 | DT | 2 | PI | 0.3128266430391681 | 33.82393943208756 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\DT_permutation_sensitivity.csv |
| regression | AC | fs2 | DT | 3 | Ucs_class | 0.0 | 0.0 | G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report\permutation_sensitivity\AC\fs2\DT_permutation_sensitivity.csv |

_Only the first 30 rows are shown here; see the CSV files for the complete table._

## Reproducibility

- Python: `3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.19045-SP0`
- Regression reports: `G:\Researching\GT_GL_MLops\GTFS_MLops_2026\regression_report`
- Classification reports: `G:\Researching\GT_GL_MLops\GTFS_MLops_2026\classification_report`

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

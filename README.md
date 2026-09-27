# Deep-Learning-Assignment-1

## Home Credit Default Risk - короткий підсумок

Ціль: передбачити ймовірність дефолту по кредиту (метрика - ROC-AUC, PR-AUC як допоміжна через дисбаланс класів). Дані: `application_train/test` + допоміжні таблиці `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`.

### Датасет

- Train: 307 511 рядків × 122 колонки, Test: 48 744 × 121.
- Таргет сильно незбалансований: **8.07% дефолтів** проти 91.93% без дефолту.
- Типи колонок: 65 float64, 41 int64, 16 object (категоріальні).

### Пропущені та неправильні значення

- 67 із 122 колонок мають пропуски, разом ~23-24% усіх комірок. 41 фіча має >50% пропусків (переважно характеристики житла: `COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, `LIVINGAPARTMENTS_*` - ~68-70% NaN).
- `DAYS_EMPLOYED` містить аномальне значення-заглушку **365243** (замість реального стажу) - це явно код "не працює/пенсіонер", а не пропуск; після заміни на NaN розподіл стає адекватним.
- У категоріальних колонках зустрічається код `XNA` як прихований пропуск: `CODE_GENDER` (4 рядки), `ORGANIZATION_TYPE` (55 374 рядки).
- Розподіл пропусків між train і test майже ідентичний (різниця <1.5 пунктів по всіх колонках) - це добре, ознака відсутності data leakage.

![Пропущені значення](images/missing_values.png)

### Розподіли та залежність від таргету

- Дефолт частіше трапляється серед молодших клієнтів: **12.3%** у групі 18-25 років проти **3.7%** у 65+ - вік один із найсильніших предикторів.
- Дохід і сума кредиту мають довгий правий хвіст (типові для грошових сум логнормальні розподіли), молодші/дефолтні позичальники частіше беруть менші кредити.
- `EXT_SOURCE_1/2/3` (зовнішні скорингові бали) - найсильніші за модулем кореляції з таргетом (від -0.16 до -0.18) і мають помітно різні щільності для класів 0 і 1.
- IQR-аналіз показує багато "викидів" у бінарних/рейтингових колонках (`REGION_RATING_CLIENT`, `FLAG_*`) - це артефакт методу на дискретних змінних, а не справжні аномалії.

![Розподіли ключових фіч за таргетом](images/distributions_by_target.png)

### Залежність фічей одна від одної (мультиколінеарність)

- Знайдено **47 пар фічей з r > 0.95** - фактично дублікати (наприклад `AMT_CREDIT` <-> `AMT_GOODS_PRICE`: 0.987, `DAYS_BIRTH` <-> `AGE`: -1.0, три групи `_AVG/_MODE/_MEDI` для однієї й тієї ж характеристики житла). Варто лишати по одній фічі з кожної пари, але оперуючи логікою, моожна і не виключати.
- Дані з `bureau` (кредитна історія) додають сигнал: клієнти з простроченнями (`HAS_DELINQUENCY`) мають вищий дефолт (15.9% проти 8.0%), але сирий лічильник попередніх кредитів майже не корелює з таргетом (-0.01).

![Кореляції та надлишковість фічей](images/correlation_redundancy.png)

## Validation

### Adversarial validation

There was done Adversarial validation between the transformed train and transformed test set. Those transformations are can be seen in ml.ipynb, like feature engineering and handling missing values. 

So mainly, there are some key drivers for train/test mismatch "categorical__NAME_CONTRACT_TYPE_Cash loans" column, financial scale features (AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE), external scores and temporal Features (EXT_SOURCE_1, YEARS_SINCE_ID_PUBLISH, BUREAU_DAYS_SINCE_LAST_LOAN). That shift distribution can explain why the diffuculty of competition and why the maxium ROC AUC is near 0.8. 

### Deep learing validation

Validation evaluation was done by calculating ROC AUC for validation set that were get by stratified split 80% / 20%. As I wrote above the test has other distribution and it is different from our validation(train) distribution that's why we get pretty optimistic evaluation

Kaggle private score / local validation

Model1: 0.71661 / 0.732

Model2: 0.72407 / 0.738

Model3: 0.72489 / 0.739

Model4: 0.74725 / 0.756

Each model is described Deep learing model section.

## Deep Learning Model

### Architecture

The model is a deep MLP designed for tabular binary classification,
built with custom PyTorch components and trained via PyTorch Lightning.

Input features (104/123) pass through a `FeatureGatingLayer` first, which
applies a learned sigmoid gate per feature — allowing the model to
suppress irrelevant inputs before any linear transformation. This is
followed by four blocks of `MyLinearLayer → MyBatchNorm → LeakyReLU → 
Dropout`, progressively reducing dimensionality from 104 → 52 → 26 → 
13 → 6, before a final `Linear(6 → 1)` output layer.

**Activation:** LeakyReLU(0.1) was chosen over ReLU to avoid zero 
gradients on negative inputs (dying neuron problem).

**Regularization:** Dropout(0.2) was applied after each block to 
reduce overfitting, alongside BatchNorm for training stability and 
gradient clipping (max_norm=1.0) to prevent exploding gradients.

**Layer sizes** follow a progressive halving strategy (104→52→26→13→6),
gradually compressing the feature space into a compact representation
before the final prediction.

---

### Custom Components

| Component | Type | Key Detail |
|-----------|------|------------|
| `MyLinearLayer` | Reimplementation | Default PyTorch weigh intialization |
| `FeatureGatingLayer` | Original | LeCun init, sigmoid gate |
| `MyBatchNorm` | Reimplementation | Manual running stats |
| `MyAdam` | Optimizer | 1st/2nd moment + bias correction |

---

### Training Configuration

| Setting | Value |
|---------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | MyAdam (lr=1e-3) |
| Scheduler | CosineAnnealingLR (T_max=10) |
| Batch size | 1024 |
| Early stopping | patience=5, monitor=val_auroc |
| Gradient clipping | max_norm=1.0 |

---

### Results

Four models were trained with progressive improvements:

| Model | Local Val AUROC | Kaggle Private AUC |
|-------|-----------------|--------------------|
| Model 1 | 0.732 | 0.71661 |
| Model 2 | 0.738 | 0.72407 |
| Model 3 | 0.739 | 0.72489 |
| Model 4 | 0.756 | 0.74725 |

model1 - only numeric data from train.csv 10 epoches
model2 - only numeric data from train.csv, added one layer(to model1) and changed dropout to 0.3
model3 - only numeric data from train.csv, added one layer(to model1) and changed dropout to 0.1
model4 - transformed training df and test df used from ml pipeline, added one layer(to model1) and changed dropout to 0.1

Here you can see all charts with metrics: https://wandb.ai/ilukianets-kyiv-school-of-economics/Deep%20learning%20assignment%201/workspace?nw=nwuserilukianets

---

### Conclusion

The best model achieved a local validation AUROC of **0.756** and a 
Kaggle private score of **0.747**, compared to the XGBoost baseline 
of **0.7706**. The neural network performs reasonably well but does not 
yet surpass the tree-based baseline, which is a common finding on 
tabular datasets where gradient boosting tends to dominate.

The gap between local validation (0.756) and Kaggle private score (0.747)
suggests mild overfitting to the validation set. A natural next step
would be adding more layers and depth to the network, alongside
stronger regularization and additional feature engineering from the
remaining auxiliary tables (credit card balances, installment payments).

### Ensembling

### Моделі: XGBoost + CatBoost + LogisticRegression (ансамбль)

- **XGBoost** (5-fold CV, з повним пайплайном препроцесингу вбудованим у sklearn `Pipeline`) - єдина модель, яка реально навчилась: OOF ROC-AUC = **0.761**.
- **Logistic Regression** і **CatBoost** у Part 2 отримують на вхід "сирий" `X_train_val` без кодування категоріальних колонок. Обидві моделі впали з помилкою `could not convert string to float: 'Cash loans'`, і в коді це "тихо" підмінюється заглушкою (LR -> випадкові числа, CatBoost -> константа 0.5 для кожного фолда). Формальний AUC CatBoost = **0.5000** (точно рандом), LogReg = **0.4953** - тобто обидві моделі по суті нічого не передбачають, а падіння прикрите try/except.
- Через це **Simple Average (0.581)** і **Weighted Average (0.658)** істотно гірші за один XGBoost, бо усереднюють корисний сигнал із двома шумовими джерелами. **Stacking (0.765)** не сильно рятує ситуацію лише тому, тому-що мета-модель (логрегресія на OOF-предиктах) навчилась ігнорувати шумові колонки LR/CatBoost і фактично копіює XGBoost, лиш іноді посилаючись на інші моделі.
- Висновок: заявлений "ансамбль трьох моделей" зараз *de facto* є одним XGBoost - щоб LR/CatBoost дали реальний внесок, потрібно прогнати їх через той самий `ColumnTransformer` (one-hot/impute), що й XGBoost, або для CatBoost - явно передати `cat_features`.

![Порівняння моделей і ROC-криві](images/model_comparison.png)

## Результати базових ансамблів з kaggle:

- Першою була спроба подивитися, чи буде працювати soft-voting навіть при погано заданих параметрів для catboost i LR. Виявилося - ні.
- Другою була ідея подивитися, який взагалі score можна отримати при звичайному\трохи зміненому xgboost, і який власне baseline. Результат - 0.763
-Третьою була спробу вже реального stacking, хоча й очевидно що при погано заданих компонентах, результат буде не сильно краще звичайного xgboost, однак незважаючи на погано задані параметри, якийсь результат це все ж дало, а саме 0.765
![Результати ансамблів](images/score_of_basic_ensembling.png)

## Soft voting

Soft voting of xgboost from ml.ipynb and deep learning model from featured-engineered-dl-model.ipynb got such result on Kaggle:

<img width="980" height="103" alt="Знімок екрана 2026-09-27 о 22 45 09" src="https://github.com/user-attachments/assets/378115af-3d0d-4ef1-be7f-2ef63c091806" />

Basically, it was done by finding the mean of two predictions files that xgboost and deep learning models made. From ensemble perspective that's a soft voting with 0.5 trust coefficients. In the end, we got something in the middle between xgboost score and deep learning model

### Файли

- `basic-data-exploration-dl.ipynb` - EDA: пропуски, викиди, кореляції, мульти-таблична аналітика.
- `dl-ensemble-model-stacking.ipynb` - пайплайн XGBoost + спроба CatBoost/LogReg + ансамблювання (Part 1-3).


### AI-usage links:
Mykola Utkin: https://docs.google.com/document/d/1ZLfL9IkIUjAuPmhv_6ddnlLEch66RpQDP84bpFKz8WQ/edit?usp=sharing - I dont know how to normally share gemini chat history

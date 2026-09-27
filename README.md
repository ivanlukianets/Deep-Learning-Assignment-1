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

### Ensembling

### Моделі: XGBoost + CatBoost + LogisticRegression (ансамбль)

- **XGBoost** (5-fold CV, з повним пайплайном препроцесингу вбудованим у sklearn `Pipeline`) - єдина модель, яка реально навчилась: OOF ROC-AUC = **0.761**.
- **Logistic Regression** і **CatBoost** у Part 2 отримують на вхід "сирий" `X_train_val` без кодування категоріальних колонок. Обидві моделі впали з помилкою `could not convert string to float: 'Cash loans'`, і в коді це "тихо" підмінюється заглушкою (LR -> випадкові числа, CatBoost -> константа 0.5 для кожного фолда). Формальний AUC CatBoost = **0.5000** (точно рандом), LogReg = **0.4953** - тобто обидві моделі по суті нічого не передбачають, а падіння прикрите try/except.
- Через це **Simple Average (0.581)** і **Weighted Average (0.658)** істотно гірші за один XGBoost, бо усереднюють корисний сигнал із двома шумовими джерелами. **Stacking (0.765)** не сильно рятує ситуацію лише тому, тому-що мета-модель (логрегресія на OOF-предиктах) навчилась ігнорувати шумові колонки LR/CatBoost і фактично копіює XGBoost, лиш іноді посилаючись на інші моделі.
- Висновок: заявлений "ансамбль трьох моделей" зараз *de facto* є одним XGBoost - щоб LR/CatBoost дали реальний внесок, потрібно прогнати їх через той самий `ColumnTransformer` (one-hot/impute), що й XGBoost, або для CatBoost - явно передати `cat_features`.

![Порівняння моделей і ROC-криві](images/model_comparison.png)

## Soft voting

Soft voting of xgboost from ml.ipynb and deep learning model from featured-engineered-dl-model.ipynb got such result on Kaggle:

<img width="980" height="103" alt="Знімок екрана 2026-09-27 о 22 45 09" src="https://github.com/user-attachments/assets/378115af-3d0d-4ef1-be7f-2ef63c091806" />

Basically, it was done by finding the mean of two predictions files that xgboost and deep learning models made. From ensemble perspective that's a soft voting with 0.5 trust coefficients. In the end, we got something in the middle between xgboost score and deep learning model

### Файли

- `basic-data-exploration-dl.ipynb` - EDA: пропуски, викиди, кореляції, мульти-таблична аналітика.
- `dl-ensemble-model-stacking.ipynb` - пайплайн XGBoost + спроба CatBoost/LogReg + ансамблювання (Part 1-3).

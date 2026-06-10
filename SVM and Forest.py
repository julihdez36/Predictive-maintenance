"""
Pipeline: SVM (Kernel Gaussiano) + Random Forest
Dataset: burned_transformers (clasificación binaria)
Estrategias de balanceo: SMOTE (sobremuestreo) y RandomUnderSampler (submuestreo)
Cross-Validation: StratifiedKFold (5 folds)
"""

# ─────────────────────────────────────────────
# 0. LIBRERÍAS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, make_scorer,
    f1_score, roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


df = pd.read_csv('data/df_entrenamiento_final.csv')
df.columns


# ─────────────────────────────────────────────
# 1. DEFINICIÓN DE FEATURES Y TARGET
# ─────────────────────────────────────────────
FEATURES = [
    'location', 'power', 'self_protection',
    'avg_earth_ddt', 'max_earth_ddt',
    'burning_rate', 'criticality_ceramics',
    'removable_connectors', 'client_type',
    'num_users', 'eens_kwh',
    'installation_type', 'air_network',
    'circuit_queue', 'network_km_lt', 'year'
]
TARGET = 'burned_transformers'

CATEGORICAL_COLS = ['client_type', 'installation_type']
NUMERICAL_COLS   = [c for c in FEATURES if c not in CATEGORICAL_COLS]


df.info()

plt.hist(df.eens_kwh);

sns.kdeplot(df.eens_kwh, fill =True);



# ─────────────────────────────────────────────
# 2. PREPROCESAMIENTO (One-Hot + Escalado)
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
    ]
)


# ─────────────────────────────────────────────
# 3. CROSS-VALIDATION
#    31 746 muestras → 5 folds estratificados
#    (~6 350 muestras por fold; test ~6 350)
# ─────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'f1'       : make_scorer(f1_score),
    'roc_auc'  : make_scorer(roc_auc_score)
}


# ─────────────────────────────────────────────
# 4. PIPELINES CON BALANCEO
# ─────────────────────────────────────────────

def build_pipeline(model, balancer):
    """Construye un ImbPipeline: preprocesamiento → balanceo → modelo."""
    return ImbPipeline(steps=[
        ('prep',     preprocessor),
        ('balance',  balancer),
        ('model',    model)
    ])


# — Modelos base —
svm_model = SVC(
    kernel='rbf',          # Kernel gaussiano
    C=1.0,
    gamma='scale',
    probability=True,      # Necesario para ROC-AUC
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',   # Peso interno como capa extra de robustez
    random_state=42,
    n_jobs=-1
)

# — Balanceadores —
# smote = SMOTE(random_state=42)
rus   = RandomUnderSampler(random_state=42)

# — Cuatro combinaciones —
pipelines = {
    "SVM  + RU"    : build_pipeline(SVC(kernel='rbf', C=1.0, gamma='scale',
                                        probability=True, random_state=42), RandomUnderSampler(random_state=42)),
    "RF   + RU"    : build_pipeline(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                            random_state=42, n_jobs=-1), RandomUnderSampler(random_state=42)),
}


# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO CON CROSS-VALIDATION
# ─────────────────────────────────────────────
X = df[FEATURES].copy()
y = df[TARGET].copy()

print("=" * 65)
print(f"  Dataset: {len(df):,} filas | Positivos: {y.sum():,} ({y.mean()*100:.1f}%)")
print("=" * 65)

cv_results = {}

for name, pipe in pipelines.items():
    print(f"\n▶  Evaluando: {name}")
    results = cross_validate(
        pipe, X, y,
        cv=CV,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1 if 'RF' in name else 1  # SVM no paraleliza bien aquí
    )
    cv_results[name] = results
    print(f"   F1       : {results['test_f1'].mean():.4f}  ±  {results['test_f1'].std():.4f}")
    print(f"   ROC-AUC  : {results['test_roc_auc'].mean():.4f}  ±  {results['test_roc_auc'].std():.4f}")


# ─────────────────────────────────────────────
# 6. RESUMEN COMPARATIVO
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN CROSS-VALIDATION (media ± desviación)")
print("=" * 65)
print(f"{'Modelo':<18} {'F1 (mean)':>10} {'F1 (std)':>9} {'AUC (mean)':>11} {'AUC (std)':>9}")
print("-" * 65)

for name, res in cv_results.items():
    print(
        f"{name:<18} "
        f"{res['test_f1'].mean():>10.4f} "
        f"{res['test_f1'].std():>9.4f} "
        f"{res['test_roc_auc'].mean():>11.4f} "
        f"{res['test_roc_auc'].std():>9.4f}"
    )


# ─────────────────────────────────────────────
# 7. ENTRENAMIENTO FINAL EN TODO EL SET
#    (para reporte y matriz de confusión)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EVALUACIÓN FINAL (último fold de CV como test)")
print("=" * 65)

# Usamos el último fold del CV para tener un set de test "limpio"
train_idx, test_idx = list(CV.split(X, y))[-1]
X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

fig, axes = plt.subplots(2, 2, figsize=(20, 9))
fig.suptitle("Reporte Final — Último Fold CV\n(SVM RBF + Random Forest  |  SMOTE vs RandomUnderSampler)",
             fontsize=13, fontweight='bold', y=1.01)

for col_idx, (name, pipe) in enumerate(pipelines.items()):
    pipe.fit(X_train_cv, y_train_cv)
    y_pred = pipe.predict(X_test_cv)

    # — Reporte de clasificación —
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_test_cv, y_pred,
                                 target_names=['No quemado', 'Quemado']))

    # — Matriz de confusión —
    ax_cm = axes[0, col_idx]
    cm = confusion_matrix(y_test_cv, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No quemado', 'Quemado'])
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title(f"{name}\nMatriz de Confusión", fontsize=10, fontweight='bold')
    ax_cm.set_xlabel("Predicho")
    ax_cm.set_ylabel("Real")

    # — Barras de métricas —
    ax_bar = axes[1, col_idx]
    report_dict = classification_report(y_test_cv, y_pred,
                                         target_names=['No quemado', 'Quemado'],
                                         output_dict=True)
    clases    = ['No quemado', 'Quemado']
    precision = [report_dict[c]['precision'] for c in clases]
    recall    = [report_dict[c]['recall']    for c in clases]
    f1        = [report_dict[c]['f1-score']  for c in clases]

    x_pos = np.arange(len(clases))
    width = 0.25
    ax_bar.bar(x_pos - width, precision, width, label='Precision', color='#4C72B0')
    ax_bar.bar(x_pos,         recall,    width, label='Recall',    color='#55A868')
    ax_bar.bar(x_pos + width, f1,        width, label='F1-score',  color='#C44E52')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(clases, fontsize=9)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Métricas por clase", fontsize=9)
    ax_bar.legend(fontsize=7)
    ax_bar.set_ylabel("Valor")
    for rect in ax_bar.patches:
        h = rect.get_height()
        ax_bar.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha='center', va='bottom', fontsize=6.5)

plt.tight_layout()
# plt.savefig("reporte_modelos_transformadores.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅  Gráfico guardado como 'reporte_modelos_transformadores.png'")
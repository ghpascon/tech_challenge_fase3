from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from scipy.stats import pointbiserialr
from imblearn.over_sampling import SMOTE

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Crédito - Análise & Predição", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/pichau/Documents/GitHub/tech_challenge_fase3/model_pipeline/data/credit.csv", encoding="utf-8")
    df.columns = df.columns.str.strip()          
    df.columns = df.columns.str.lower()          
    df.columns = df.columns.str.replace(" ", "_")
    return df
@st.cache_data
def tratar_database(data):
    conta_corrente_map = {
        'no checking account': 0,
        '< 0 DM': 0
    }

    data['conta_corrente'] = data['conta_corrente'].map(lambda x: conta_corrente_map.get(x, 1))

    credit_history_map = {
        'critical account/ other credits existing (not at this bank)': 0,
        'delay in paying off in the past': 0,
        'existing credits paid back duly till now': 1,
        'no credits taken/ all credits paid back duly': 1,
        'all credits at this bank paid back duly': 1
    }

    data['historico_credito'] = data['historico_credito'].map(credit_history_map)

    purpose_map = {
        'domestic appliances': 0,
        '(vacation - does not exist?)': 0,
        'radio/television': 0,
        'car (new)': 1,
        'car (used)': 1,
        'business': 1,
        'repairs': 0,
        'education': 1,
        'furniture/equipment': 1,
        'retraining': 1
    }

    data['proposito_emprestimo'] = data['proposito_emprestimo'].map(purpose_map)

    savings_map = {
        'unknown/ no savings account': 0,
        '... < 100 DM': 0,
        '100 <= ... < 500 DM': 0,
        '500 <= ... < 1000 DM ': 1,
        '.. >= 1000 DM ': 1
    }

    data['reserva_cc'] = data['reserva_cc'].map(savings_map)


    employment_map = {
        'unemployed': 0,
        '... < 1 year ': 0,
        '1 <= ... < 4 years': 1,
        '4 <= ... < 7 years': 1,
        '.. >= 7 years': 1
    }

    data['tempo_emprego_atual'] = data['tempo_emprego_atual'].map(employment_map)

    data['outros_fiadores'] = data['outros_fiadores'].map(lambda x: 0 if x == 'none' else 1)

    property_map = {
        'unknown / no property': 0,
        'if not A121 : building society savings agreement/ life insurance': 1,
        'if not A121/A122 : car or other, not in attribute 6': 1,
        'real estate': 2
    }

    data['propriedade'] = data['propriedade'].map(property_map)

    data['outros_planos_financiamento'] = data['outros_planos_financiamento'].map(lambda x: 0 if x == 'none' else 1)

    data['tipo_residencia'] = data['tipo_residencia'].map(lambda x: 1 if x in ['own', 'for free'] else 0)

    status_emprego_map = {
        'unemployed/ unskilled - non-resident': 0,
        'unskilled - resident': 0,
        'skilled employee / official': 1,
        'management/ self-employed/ highly qualified employee/ officer': 2
    }

    data['status_emprego'] = data['status_emprego'].map(status_emprego_map)

    data['telefone'] = data['telefone'].map(lambda x: 0 if x == 'none' else 1)

    data['trabalhador_estrangeiro'] = data['trabalhador_estrangeiro'].map(lambda x: 0 if x == 'no' else 1)

    dm_to_euro = 1 / 1.95583  # 1 Euro = 1,95583 DM
    euro_to_brl = 6.37  # 1 Euro = 6,37 BRL
    data['valor_emprestimo'] = data['valor_emprestimo'].apply(lambda x: x * dm_to_euro * euro_to_brl)
    
    data['prazo_emprestimo_anos'] = data['prazo_emprestimo_meses'].apply(lambda x: int(x/12))
    data.drop('prazo_emprestimo_meses', axis=1, inplace=True)

    bins = list(range(0, 101, 10))

    data['faixa_idade'] = pd.cut(data['idade'], bins=bins, right=False, labels=False)
    data.drop(columns=['idade'], inplace=True, axis=1)

    return data

def evaluate_model(model, x_train, y_train, x_test, y_test, cv_splits=10):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc_test = accuracy_score(y_test, pred)
    f1_test = f1_score(y_test, pred, average="weighted")

    #RocCurveDisplay.from_estimator(model, x_train, y_train)
    if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
    else:
            y_proba = model.decision_function(x_test)

    # Calcular curva ROC e AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar curva ROC
    plt.plot(fpr, tpr, label=f'{type(model).__name__} (AUC = {roc_auc:.3f})', linewidth=2)
    #plt.plot(fpr, tpr, label=f'(AUC = {roc_auc:.3f})', linewidth=2)

    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])


    return {
        "Modelo": type(model).__name__,
        "CV_Mean": cv_scores.mean(),
        "CV_Std": cv_scores.std(),
        "Accuracy_Test": acc_test,
        "F1_Test": f1_test
    }

def reduzir_dimensionalidade(data):

    # Supondo que 'data' seja seu DataFrame e 'default' seja o target binário
    features = data.drop(columns=['default']).columns
    correlations = []

    for col in features:
        corr, _ = pointbiserialr(data['default'], data[col])
        correlations.append(corr)

    # Criar DataFrame com correlações
    corr_df = pd.DataFrame({'feature': features, 'correlation_with_target': correlations})
    corr_df = corr_df.sort_values(by='correlation_with_target', key=abs, ascending=False)  # ordena pelo valor absoluto

    # Selecionar as top 10 features mais influentes
    top_n = 10
    top_features = corr_df['feature'].iloc[:top_n].tolist()

    # Ordenar alfabeticamente
    top_features_sorted = sorted(top_features)

    # Criar um novo DataFrame apenas com as top features + target
    data_top = data[top_features_sorted + ['default']]


    # Separar features e target
    X = data_top.drop(columns=['default'])
    y = data_top['default']

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Criar DataFrame balanceado
    data_resampled = pd.DataFrame(X_res, columns=X.columns)
    data_resampled['default'] = y_res

    # Criar pipeline de pré-processamento (sem modelo ainda)
    preprocessor = Pipeline([
        ('scaler', StandardScaler()),   # normaliza os dados
    ])

    # Separar features e target
    X = data_resampled.drop(columns=['default'])
    y = data_resampled['default']

    # Ajustar pré-processador aos dados
    X_transformed = preprocessor.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.1)


    return x_train, x_test, y_train, y_test

def gerar_grafico_curva_roc(x_train, y_train, x_test, y_test):

    random_state = 1375

    modelos = [
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state),
        RandomForestClassifier(n_estimators=21, max_depth=10, random_state=random_state),
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=random_state),
        SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=random_state),
        KNeighborsClassifier(n_neighbors=7),
        DecisionTreeClassifier(max_depth=6, random_state=random_state),
        GaussianNB()
    ]

    # Avaliar todos e guardar resultados
    resultados = []
    for m in modelos:
        resultados.append(evaluate_model(m, x_train, y_train, x_test, y_test))

df = load_data()

st.set_page_config(layout="wide")

st.title("📊 Dashboard de Análise de Empréstimos")

# ==============================
# Top KPIs
# ==============================
total_clientes = len(df)
percent_default = (df["default"] == 0).mean() * 100
media_valor_emprestimo = df["valor_emprestimo"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Clientes", total_clientes)
col2.metric("Percentual de Aprovação", f"{percent_default:.2f}%")
col3.metric("Média do Valor do Empréstimo", f"${media_valor_emprestimo:,.2f}")

# ==============================
# Filtros interativos
# ==============================
st.sidebar.header("Filtros")
faixa_idade = st.sidebar.slider("Idade", int(df["idade"].min()), int(df["idade"].max()), (int(df["idade"].min()), int(df["idade"].max())))
sexo_est_civil = st.sidebar.multiselect("Sexo / Estado Civil", df["sexo_est_civil"].unique())
tipo_residencia = st.sidebar.multiselect("Tipo de Residência", df["tipo_residencia"].unique())
proposito_emprestimo = st.sidebar.multiselect("Propósito do Empréstimo", df["proposito_emprestimo"].unique())
historico_credito = st.sidebar.multiselect("Histórico de Crédito", df["historico_credito"].unique())

df_filtrado = df.copy()
df_filtrado = df_filtrado[(df_filtrado["idade"] >= faixa_idade[0]) & (df_filtrado["idade"] <= faixa_idade[1])]
if sexo_est_civil:
    df_filtrado = df_filtrado[df_filtrado["sexo_est_civil"].isin(sexo_est_civil)]
if tipo_residencia:
    df_filtrado = df_filtrado[df_filtrado["tipo_residencia"].isin(tipo_residencia)]
if proposito_emprestimo:
    df_filtrado = df_filtrado[df_filtrado["proposito_emprestimo"].isin(proposito_emprestimo)]
if historico_credito:
    df_filtrado = df_filtrado[df_filtrado["historico_credito"].isin(historico_credito)]

# ==============================
# Gráficos – Perfil do Cliente
# ==============================
st.header("👤 Perfil do Cliente")

# Idade

df_filtrado['status'] = df_filtrado['default'].map({0: 'Aprovado', 1: 'Reprovado'})

st.subheader("Distribuição de Idade")
fig_idade = px.histogram(df_filtrado, x="idade", color="status", barmode="group")
st.plotly_chart(fig_idade, use_container_width=True)

# Sexo / Estado Civil
st.subheader("Aprovação por Sexo / Estado Civil")
fig_sexo = px.bar(df_filtrado.groupby(["sexo_est_civil", "status"]).size().reset_index(name="count"),
                  x="sexo_est_civil", y="count", color="status", barmode="group")
st.plotly_chart(fig_sexo, use_container_width=True)

# Tipo de residência e propriedade
st.subheader("Propriedade e Tipo de Residência")
fig_prop = px.pie(df_filtrado, names="propriedade", title="Propriedade dos Clientes")
st.plotly_chart(fig_prop, use_container_width=True)

# ==============================
# Gráficos – Empréstimos e Finanças
# ==============================
st.header("💰 Empréstimos e Finanças")

# Valor do empréstimo
st.subheader("Distribuição do Valor do Empréstimo")
fig_valor = px.box(df_filtrado, x="status", y="valor_emprestimo", color="status")
st.plotly_chart(fig_valor, use_container_width=True)

# Prazo do empréstimo
st.subheader("Prazo do Empréstimo x Aprovação")
fig_prazo = px.histogram(df_filtrado, x="prazo_emprestimo_meses", color="status", barmode="overlay")
st.plotly_chart(fig_prazo, use_container_width=True)

# Taxa comprometida do salário
st.subheader("Taxa Comprometida do Salário x Aprovação")
fig_taxa = px.box(df_filtrado, x="status", y="taxa_comp_salario", color="status")
st.plotly_chart(fig_taxa, use_container_width=True)

# ==============================
# Gráficos – Histórico de Crédito
# ==============================
st.header("📈 Histórico de Crédito")

# Histórico de crédito
st.subheader("Taxa de Aprovação por Histórico de Crédito")
taxa = df_filtrado.groupby("historico_credito")["default"].mean().reset_index()
fig_hist = px.bar(taxa, x="historico_credito", y="default", title="Taxa de Aprovação (%)", text_auto=True)
st.plotly_chart(fig_hist, use_container_width=True)

# Propósito do empréstimo
st.subheader("Aprovação por Propósito do Empréstimo")
fig_prop_emprest = px.bar(df_filtrado.groupby(["proposito_emprestimo", "default"]).size().reset_index(name="count"),
                          x="proposito_emprestimo", y="count", color="default", barmode="group")
st.plotly_chart(fig_prop_emprest, use_container_width=True)

# ==============================
# Gráficos – Visões Avançadas
# ==============================
st.header("🔍 Visões Avançadas de Risco")

# Default vs Número de Dependentes
fig_dependentes = px.box(df_filtrado, x="status", y="n_dependentes", color="status", points="all")
st.subheader("Aprovação vs Número de Dependentes")
st.plotly_chart(fig_dependentes, use_container_width=True)

# Default vs Número de Créditos no Banco
fig_creditos = px.box(df_filtrado, x="status", y="n_creditos_banco", color="status", points="all")
st.subheader("Aprovação vs Número de Créditos no Banco")
st.plotly_chart(fig_creditos, use_container_width=True)

# Default vs Status e Tempo de Emprego
fig_emprego = px.box(df_filtrado, x="status_emprego", y="tempo_emprego_atual", color="status", points="all")
st.subheader("Tempo de Emprego Atual por Status e Aprovação")
st.plotly_chart(fig_emprego, use_container_width=True)


df_tratado = tratar_database(df_filtrado)

fig, ax = plt.subplots(figsize=(8, 6))
st.subheader("Mapa de calor - correlação entre as variáveis mais significativas")
correlacionados = ["default", "prazo_emprestimo_anos", "historico_credito", "valor_emprestimo", "propriedade", "outros_planos_financiamento"]
sns.heatmap(
    df_tratado[correlacionados].corr(),
    annot=True,
    center=0,
    square=True,  # deixa proporcional
    cbar_kws={"shrink": 0.8},  # barra menor
    ax=ax
)
ax.set_title("Heatmap de Correlação", fontsize=14, fontweight="bold")
st.pyplot(fig, use_container_width=True)


X_train, X_test, y_train, y_test = reduzir_dimensionalidade(df_tratado.drop(columns=['status','sexo_est_civil']))

# Plotar curvas ROC
st.subheader("Curvas ROC - Comparação de Modelos")
plt.figure(figsize=(10, 8))

gerar_grafico_curva_roc(X_train, y_train, X_test, y_test)
plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
plt.legend(loc="lower right")
plt.tight_layout()
st.pyplot(plt.gcf())


# ==============================
# Tabela interativa de dados
# ==============================
st.header("📋 Dados Filtrados")
st.dataframe(df_filtrado)


# -----------------------------
# Predição
# -----------------------------
st.header("🔮 Predição de Aprovação de Empréstimo")

with st.form("prediction_form"):
    valor_emprestimo = st.number_input("Valor do empréstimo (R$)", min_value=500, max_value=100000, step=500)
    prazo = st.slider("Prazo do empréstimo (anos)", 1, 10, 2)
    idade = st.number_input("Idade", min_value=18, max_value=100, step=1)

    propriedade = st.selectbox("Propriedade", [0, 1, 2])
    tempo_emprego = st.selectbox("Tempo no emprego atual", [0, 1])
    reserva_cc = st.selectbox("Reserva em conta corrente", [0, 1])

    # Possui outros empréstimos
    outros_emprestimos_opts = {"Não": 0, "Sim": 1}
    outros_emprestimos = st.selectbox(
        "Possui outros empréstimos?",
        options=list(outros_emprestimos_opts.keys())
    )

    # Histórico de crédito
    historico_opts = {
        "Crítico / Atrasos": 0,
        "Bom histórico": 1
    }
    historico_credito = st.selectbox(
        "Histórico de crédito",
        options=list(historico_opts.keys())
    )

    # Tipo de residência
    residencia_opts = {"Alugada": 0, "Própria": 1}
    tipo_residencia = st.selectbox(
        "Tipo de residência",
        options=list(residencia_opts.keys())
    )

    # Conta corrente
    cc_opts = {"Sem conta / negativa": 0, "Conta positiva": 1}
    conta_corrente = st.selectbox(
        "Conta corrente",
        options=list(cc_opts.keys())
    )

    submitted = st.form_submit_button("Enviar para predição")

if submitted:
    payload = {
        "valor_emprestimo": valor_emprestimo,
        "prazo_emprestimo": prazo,
        "idade": idade,
        "outros_emprestimos": outros_emprestimos_opts[outros_emprestimos],
        "historico_credito": historico_opts[historico_credito],
        "propriedade": propriedade,
        "tempo_emprego": tempo_emprego,
        "reserva_cc": reserva_cc,
        "tipo_residencia": residencia_opts[tipo_residencia],
        "conta_corrente": cc_opts[conta_corrente],
    }

    response = requests.post(f"{API_URL}/api/pred/pred_loan_classification", json=payload)

    if response.status_code == 200:
        result = response.json()
        if result["pred"] == 0:
            st.success("✅ Empréstimo aprovado!")
        else:
            st.error("❌ Empréstimo negado.")
    else:
        st.error("Erro ao conectar com a API.")



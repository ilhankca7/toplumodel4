import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("C:/Users/ilhan/Downloads/customer.csv")
veri=data.copy()
veri=veri.drop(columns="customerID",axis=1)
turkce_sutunlar = {
    "gender": "Cinsiyet",
    "SeniorCitizen": "65 Yaş Üstü",
    "Partner": "Medeni Durum",
    "Dependents": "Bakma Sorumluluğu",
    "tenure": "Müşteri Olma Süresi(Ay)",
    "PhoneService": "Ev Telefonu Aboneliği",
    "MultipleLines": "Birden Fazla Abonelik Durumu",
    "InternetService": "İnternet Aboneliği",
    "OnlineSecurity": "Güvenlik Hizmeti Aboneliği",
    "OnlineBackup": "Yedekleme Hizmeti Aboneliği",
    "DeviceProtection": "Ekipman Güvenlik Aboneliği",
    "TechSupport": "Teknik Destek Aboneliği",
    "StreamingTV": "IP Tv Aboneliği",
    "StreamingMovies": "Film Aboneliği",
    "Contract": "Sözleşme Süresi",
    "PaperlessBilling": "Online Fatura(Kağıtsız)",
    "PaymentMethod": "Ödeme Şekli",
    "MonthlyCharges": "Aylık Ücret",
    "TotalCharges": "Toplam Ücret",
    "Churn": "Kayıp Durumu"
}

veri = veri.rename(columns=turkce_sutunlar)

veri["Cinsiyet"]=["Erkek" if kod=="Male" else "Kadın" for kod in veri["Cinsiyet"]]
veri["65 Yaş Üstü"]=["Evet" if kod==1 else "Hayır" for kod in veri["65 Yaş Üstü"]]
veri["Medeni Durum"]=["Evli" if kod=="Yes" else "Bekar" for kod in veri["Medeni Durum"]]                  
veri["Bakma Sorumluluğu"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Bakma Sorumluluğu"]]                  
veri["Ev Telefonu Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Ev Telefonu Aboneliği"]]
veri["Birden Fazla Abonelik Durumu"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Birden Fazla Abonelik Durumu"]]
veri["İnternet Aboneliği"]=["Yok" if kod=="No" else "Var" for kod in veri["İnternet Aboneliği"]]
veri["Güvenlik Hizmeti Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Güvenlik Hizmeti Aboneliği"]]
veri["Yedekleme Hizmeti Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Yedekleme Hizmeti Aboneliği"]]
veri["Ekipman Güvenlik Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Ekipman Güvenlik Aboneliği"]]
veri["Teknik Destek Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Teknik Destek Aboneliği"]]
veri["IP Tv Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["IP Tv Aboneliği"]]
veri["Film Aboneliği"]=["Var" if kod=="Yes" else "Yok" for kod in veri["Film Aboneliği"]]
veri["Sözleşme Süresi"] = ["1 Aylık" if kod == "Month-to-month" else "1 Yıllık" if kod == "One year" else "2 Yıllık" for kod in veri["Sözleşme Süresi"]]
veri["Online Fatura(Kağıtsız)"]=["Evet" if kod=="Yes" else "Hayır" for kod in veri["Online Fatura(Kağıtsız)"]]
veri["Ödeme Şekli"] = ["Elektronik" if kod == "Electronic check" else "Mail" if kod == "Mailed check" else "Havale" if kod == "Bank transfer (automatic)" else "Kredi Kart" for kod in veri["Ödeme Şekli"]]
veri["Kayıp Durumu"]=["Evet" if kod=="Yes" else "Hayır" for kod in veri["Kayıp Durumu"]]
veri["Toplam Ücret"]=pd.to_numeric(veri["Toplam Ücret"],errors="coerce")
veri=veri.dropna()

le=LabelEncoder()
degisken=veri.select_dtypes(include="object").columns
veri.update(veri[degisken].apply(le.fit_transform))

veri["Kayıp Durumu"]=[1 if kod==0 else 0 for kod in veri["Kayıp Durumu"]]
y=veri["Kayıp Durumu"]
X=veri.drop(columns="Kayıp Durumu",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


models=["LinearSVC","SVC","Ridge","Logistic","RandomForest","LGBM","XGBM"]
sınıflar=[LinearSVC(random_state=0),SVC(random_state=0),RidgeClassifier(random_state=0),
         LogisticRegression(random_state=0),RandomForestClassifier(random_state=0),
         LGBMClassifier(random_state=0),XGBClassifier()]


parametreler={
    models[0]:{"C":[0.1,1,10,100],"penalty":["l1","l2"]},
    models[1]:{"kernel":["linear","rbf"],"C":[0.1,2],"gamma":[0.01,0.001]},
    models[2]:{"alpha":[0.1,1.0]},
    models[3]:{"C":[0.1,1],"penalty":["l1","l2"]},
    models[4]:{"n_estimators":[1000,2000],"max_depth":[4,10],"min_samples_split":[2,5]},
    models[5]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"mx_depth":[4,10],"subsample":[0.6,0.8]},
    models[6]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"mx_depth":[4,10],"subsample":[0.6,0.8]}}

def cozum(model):
    model.fit(X_train,y_train)
    return model
    

def skor(model2):
    tahmin=cozum(model2).predict(X_test)
    acs=accuracy_score(y_test,tahmin)
    return acs*100

for i,j in zip(models,sınıflar):
    print(i)
    grid=GridSearchCV(cozum(j),parametreler[i],cv=10,n_jobs=-1)
    grid.fit(X_train,y_train)
    print(grid.best_params_)
    






















import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Dense, Input
from keras.models import Model
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


if __name__ == "__main__":
    NUM_FOLDS = 50
    SEED = 1000

    shigeria_pred1 = np.load("shigeria_pred1.npy")
    shigeria_pred2 = np.load("shigeria_pred2.npy")
    shigeria_pred3 = np.load("shigeria_pred3.npy")
    shigeria_pred4 = np.load("shigeria_pred4.npy")
    shigeria_pred5 = np.load("shigeria_pred5.npy")
    shigeria_pred6 = np.load("shigeria_pred6.npy")
    shigeria_pred7 = np.load("shigeria_pred7.npy")
    shigeria_pred8 = np.load("shigeria_pred8.npy")
    shigeria_pred9 = np.load("shigeria_pred9.npy")
    shigeria_pred10 = np.load("shigeria_pred10.npy")
    upura_pred = np.load("upura_pred.npy")
    takuoko_exp085 = np.load("takuoko_exp085.npy")
    takuoko_exp096 = np.load("takuoko_exp096.npy")
    takuoko_exp105 = np.load("takuoko_exp105.npy")
    takuoko_exp108 = np.load("takuoko_exp108.npy")
    takuoko_exp184 = np.load("takuoko_exp184.npy")
    X_train_svd = np.load("X_train_all.npy")
    X_test_svd = np.load("X_test_all.npy")
    train_idx = np.load("train_idx.npy", allow_pickle=True)

    svd1 = TruncatedSVD(n_components=3, n_iter=10, random_state=42)
    svd1.fit(X_train_svd)
    X_train_svd = svd1.transform(X_train_svd)
    X_test_svd = svd1.transform(X_test_svd)

    X_test = pd.DataFrame(
        {
            "shigeria_pred1": shigeria_pred1.reshape(-1),
            "shigeria_pred2": shigeria_pred2.reshape(-1),
            "shigeria_pred3": shigeria_pred3.reshape(-1),
            "shigeria_pred4": shigeria_pred4.reshape(-1),
            "shigeria_pred5": shigeria_pred5.reshape(-1),
            "shigeria_pred6": shigeria_pred6.reshape(-1),
            "shigeria_pred7": shigeria_pred7.reshape(-1),
            "shigeria_pred8": shigeria_pred8.reshape(-1),
            "shigeria_pred9": shigeria_pred9.reshape(-1),
            "shigeria_pred10": shigeria_pred10.reshape(-1),
            "upura": upura_pred,
            "takuoko_exp085": takuoko_exp085,
            "takuoko_exp096": takuoko_exp096,
            "takuoko_exp105": takuoko_exp105,
            "takuoko_exp108": takuoko_exp108,
            "takuoko_exp184": takuoko_exp184,
        }
    )
    X_test = pd.concat(
        [
            X_test,
            pd.DataFrame(
                X_test_svd, columns=[f"svd_{c}" for c in range(X_test_svd.shape[1])]
            ),
        ],
        axis=1,
    )

    # upura oof
    pred_val000 = pd.read_csv("../input/commonlit-oof/pred_val000.csv")

    # shigeria oof
    andrey_df = pd.read_csv("../input/commonlitstackingcsv/roberta_base_itpt.csv")
    andrey_df2 = pd.read_csv("../input/commonlitstackingcsv/attention_head_nopre.csv")
    andrey_df3 = pd.read_csv("../input/commonlitstackingcsv/attention_head_itpt.csv")
    andrey_df4 = pd.read_csv(
        "../input/d/shigeria/bayesian-commonlit/np_savetxt_andrey4.csv"
    )
    andrey_df5 = pd.read_csv("../input/commonlitstackingcsv/mean_pooling_last1.csv")
    andrey_df6 = pd.read_csv(
        "../input/commonlitstackingcsv/attention_head_cls_last3s.csv"
    )
    andrey_df7 = pd.read_csv(
        "../input/commonlitstackingcsv/mean_pooling_cls_last3s.csv"
    )
    andrey_df8 = pd.read_csv(
        "../input/commonlitstackingcsv/attention_head_cls_last4s.csv"
    )
    andrey_df9 = pd.read_csv("../input/commonlitstackingcsv/electra_large_nopre.csv")
    andrey_df10 = pd.read_csv(
        "../input/commonlitstackingcsv/attention_head_mean_pooling_cls_last3s.csv"
    )

    # takuoko oof
    pred_val085 = pd.read_csv("../input/commonlit-oof/pred_val085.csv")
    pred_val096 = pd.read_csv("../input/commonlit-oof/pred_val096.csv")
    pred_val105 = pd.read_csv("../input/commonlit-oof/pred_val105.csv")
    pred_val108 = pd.read_csv("../input/commonlit-oof/pred_val108.csv")
    pred_val184 = pd.read_csv("../input/commonoof/exp184_roberta_large_10fold.csv")
    pred_val085 = pd.merge(pred_val000[["id"]], pred_val085, on="id", how="left")
    pred_val096 = pd.merge(pred_val000[["id"]], pred_val096, on="id", how="left")
    pred_val105 = pd.merge(pred_val000[["id"]], pred_val105, on="id", how="left")
    pred_val108 = pd.merge(pred_val000[["id"]], pred_val108, on="id", how="left")
    pred_val184 = pd.merge(pred_val000[["id"]], pred_val184, on="id", how="left")

    y_train = pred_val085.true_target
    X_train = pd.DataFrame(
        {
            "shigeria_pred1": andrey_df.pred.values,
            "shigeria_pred2": andrey_df2.pred.values,
            "shigeria_pred3": andrey_df3.pred.values,
            "shigeria_pred4": andrey_df4.pred.values,
            "shigeria_pred5": andrey_df5.pred.values,
            "shigeria_pred6": andrey_df6.pred.values,
            "shigeria_pred7": andrey_df7.pred.values,
            "shigeria_pred8": andrey_df8.pred.values,
            "shigeria_pred9": andrey_df9.pred.values,
            "shigeria_pred10": andrey_df10.pred.values,
            "upura": pred_val000.pred_target.values,
            "takuoko_exp085": pred_val085.pred_target.values,
            "takuoko_exp096": pred_val096.pred_target.values,
            "takuoko_exp105": pred_val105.pred_target.values,
            "takuoko_exp108": pred_val108.pred_target.values,
            "takuoko_exp184": pred_val184.pred_target.values,
        }
    )
    X_train_svd = pd.DataFrame(
        X_train_svd, columns=[f"svd_{c}" for c in range(X_train_svd.shape[1])]
    )
    X_train_svd.index = train_idx
    X_train_svd = X_train_svd.reindex(index=pred_val085["id"]).reset_index(drop=True)
    X_train = pd.concat([X_train, X_train_svd], axis=1)

    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train)))
    y_preds_r = []
    models_r = []
    oof_train_r = np.zeros((len(X_train)))
    y_preds_e = []
    models_e = []
    oof_train_e = np.zeros((len(X_train)))
    y_preds_n = []
    models_n = []
    oof_train_n = np.zeros((len(X_train)))
    cv = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    params_r = {"alpha": 10, "random_state": 0}
    params_e = {
        "max_depth": 10,
        "min_samples_leaf": 1,
        "max_features": 0.9,
        "n_estimators": 50,
        "random_state": 0,
    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        model = ARDRegression(
            n_iter=300,
            alpha_1=4.419269430437814e-05,
            alpha_2=7.067678595336767e-08,
            lambda_1=9886.63123193296,
            lambda_2=71.70204561520023,
            verbose=True,
        )
        model.fit(X_tr, y_tr)
        oof_train[valid_index] = model.predict(X_val)
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        models.append(model)

        model_r = Ridge(**params_r)
        model_r.fit(X_tr, y_tr)
        oof_train_r[valid_index] = model_r.predict(X_val)
        y_pred_r = model_r.predict(X_test)
        y_preds_r.append(y_pred_r)
        models_r.append(model_r)

        model_e = ExtraTreesRegressor(**params_e)
        model_e.fit(X_tr, y_tr)
        oof_train_e[valid_index] = model_e.predict(X_val)
        y_pred_e = model_e.predict(X_test)
        y_preds_e.append(y_pred_e)
        models_e.append(model_e)

        inp_ = Input(shape=[X_tr.shape[1]])
        nn = Dense(50)(inp_)
        nn = BatchNormalization()(nn)
        nn = Activation("relu")(nn)
        nn = Dense(16)(nn)
        nn = BatchNormalization()(nn)
        nn = Activation("relu")(nn)
        nn = Dense(1)(nn)
        model = Model(inp_, nn)
        model.compile(optimizer="Adam", loss=root_mean_squared_error)

        file_path = "model.hdf5"
        ckpt = ModelCheckpoint(
            file_path, monitor="val_loss", verbose=0, save_best_only=True, mode="min"
        )
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        model.fit(
            X_tr,
            y_tr,
            batch_size=64,
            epochs=100,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[ckpt, early],
        )
        model.load_weights(file_path)
        oof_train_n[valid_index] = model.predict(X_val).reshape(-1)
        y_pred_n = model.predict(X_test).reshape(-1)
        y_preds_n.append(y_pred_n)

    print(mean_squared_error(oof_train, y_train, squared=False))
    y_sub = sum(y_preds) / len(y_preds)

    print(mean_squared_error(oof_train_r, y_train, squared=False))
    y_sub_r = sum(y_preds_r) / len(y_preds_r)

    print(mean_squared_error(oof_train_e, y_train, squared=False))
    y_sub_e = sum(y_preds_e) / len(y_preds_e)

    print(mean_squared_error(oof_train_n, y_train, squared=False))
    y_sub_n = sum(y_preds_n) / len(y_preds_n)

    print(
        mean_squared_error(
            (oof_train + oof_train_r + oof_train_e + oof_train_n) / 4,
            y_train,
            squared=False,
        )
    )

    submission_df = pd.read_csv(
        "../input/commonlitreadabilityprize/sample_submission.csv"
    )
    submission_df["target"] = (y_sub + y_sub_r + y_sub_e + y_sub_n) / 4
    submission_df.to_csv("submission.csv", index=False)
    print(submission_df.head())

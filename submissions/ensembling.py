import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

if __name__ == "__main__":
    NUM_FOLDS = 5
    SEED = 1000

    shigeria_pred1 = np.load("shigeria_pred1.npy")
    shigeria_pred2 = np.load("shigeria_pred2.npy")
    shigeria_pred3 = np.load("shigeria_pred3.npy")
    shigeria_pred4 = np.load("shigeria_pred4.npy")
    upura_pred = np.load("upura_pred.npy")
    takuoko_exp085 = np.load("takuoko_exp085.npy")
    takuoko_exp096 = np.load("takuoko_exp096.npy")
    takuoko_exp105 = np.load("takuoko_exp105.npy")
    takuoko_exp108 = np.load("takuoko_exp108.npy")
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
            "upura": upura_pred,
            "takuoko_exp085": takuoko_exp085,
            "takuoko_exp096": takuoko_exp096,
            "takuoko_exp105": takuoko_exp105,
            "takuoko_exp108": takuoko_exp108,
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
    # takuoko oof
    pred_val085 = pd.read_csv("../input/commonlit-oof/pred_val085.csv")
    pred_val096 = pd.read_csv("../input/commonlit-oof/pred_val096.csv")
    pred_val105 = pd.read_csv("../input/commonlit-oof/pred_val105.csv")
    pred_val108 = pd.read_csv("../input/commonlit-oof/pred_val108.csv")

    pred_val105.index = pred_val105["id"]
    pred_val105 = pred_val105.reindex(index=pred_val085["id"]).reset_index(drop=True)
    pred_val108.index = pred_val108["id"]
    pred_val108 = pred_val108.reindex(index=pred_val085["id"]).reset_index(drop=True)

    delete_idx = pred_val085[pred_val085.id == "436ce79fe"].index
    pred_val085 = pred_val085.drop(delete_idx).reset_index(drop=True)
    pred_val096 = pred_val096.drop(delete_idx).reset_index(drop=True)
    pred_val105 = pred_val105.drop(delete_idx).reset_index(drop=True)
    pred_val108 = pred_val108.drop(delete_idx).reset_index(drop=True)

    # upura oof
    pred_val000 = pd.read_csv("../input/commonlit-oof/pred_val000.csv")
    pred_val000.index = pred_val000["id"]
    pred_val000 = pred_val000.reindex(index=pred_val085["id"]).reset_index(drop=True)

    # shigeria oof
    andrey_df = pd.read_csv(
        "../input/d/shigeria/bayesian-commonlit/np_savetxt_andrey.csv", header=None
    ).values.ravel()
    andrey_df2 = pd.read_csv(
        "../input/d/shigeria/bayesian-commonlit/np_savetxt_andrey2.csv", header=None
    ).values.ravel()
    andrey_df3 = pd.read_csv(
        "../input/d/shigeria/bayesian-commonlit/np_savetxt_andrey3.csv", header=None
    ).values.ravel()
    andrey_df4 = pd.read_csv(
        "../input/d/shigeria/bayesian-commonlit/np_savetxt_andrey4.csv", header=None
    ).values.ravel()

    shigeria_val = pd.read_csv("../input/commonlit-oof/pred_val000.csv")
    shigeria_val["pred_target"] = andrey_df
    shigeria_val.index = shigeria_val["id"]
    shigeria_val = shigeria_val.reindex(index=pred_val085["id"]).reset_index(drop=True)

    shigeria_val2 = pd.read_csv("../input/commonlit-oof/pred_val000.csv")
    shigeria_val2["pred_target"] = andrey_df2
    shigeria_val2.index = shigeria_val2["id"]
    shigeria_val2 = shigeria_val2.reindex(index=pred_val085["id"]).reset_index(
        drop=True
    )

    shigeria_val3 = pd.read_csv("../input/commonlit-oof/pred_val000.csv")
    shigeria_val3["pred_target"] = andrey_df3
    shigeria_val3.index = shigeria_val3["id"]
    shigeria_val3 = shigeria_val3.reindex(index=pred_val085["id"]).reset_index(
        drop=True
    )

    shigeria_val4 = pd.read_csv("../input/commonlit-oof/pred_val000.csv")
    shigeria_val4["pred_target"] = andrey_df4
    shigeria_val4.index = shigeria_val4["id"]
    shigeria_val4 = shigeria_val4.reindex(index=pred_val085["id"]).reset_index(
        drop=True
    )

    y_train = pred_val085.true_target
    X_train = pd.DataFrame(
        {
            "shigeria_pred1": shigeria_val.pred_target.values,
            "shigeria_pred2": shigeria_val2.pred_target.values,
            "shigeria_pred3": shigeria_val3.pred_target.values,
            "shigeria_pred4": shigeria_val4.pred_target.values,
            "upura": pred_val000.pred_target.values,
            "takuoko_exp085": pred_val085.pred_target.values,
            "takuoko_exp096": pred_val096.pred_target.values,
            "takuoko_exp105": pred_val105.pred_target.values,
            "takuoko_exp108": pred_val108.pred_target.values,
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
    cv = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    params_r = {"alpha": 10, "random_state": 0}

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

    print(mean_squared_error(oof_train, y_train, squared=False))
    y_sub = sum(y_preds) / len(y_preds)

    print(mean_squared_error(oof_train_r, y_train, squared=False))
    y_sub_r = sum(y_preds_r) / len(y_preds_r)

    print(
        mean_squared_error(oof_train * 0.7 + oof_train_r * 0.3, y_train, squared=False)
    )

    submission_df = pd.read_csv(
        "../input/commonlitreadabilityprize/sample_submission.csv"
    )
    submission_df["target"] = y_sub * 0.7 + y_sub_r * 0.3
    submission_df.to_csv("submission.csv", index=False)
    print(submission_df.head())

import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def f(x):
    y_pred = 0
    for i, d in enumerate(y_preds):
        if i < len(x):
            y_pred += d * x[i]
        else:
            y_pred += d * (1 - sum(x))
    score = mean_squared_error(y_true, y_pred, squared=False)
    return score


if __name__ == '__main__':
    # takuoko
    pred_val085 = pd.read_csv('../external/commonlit-oof/pred_val085.csv')
    pred_val096 = pd.read_csv('../external/commonlit-oof/pred_val096.csv')
    pred_val105 = pd.read_csv('../external/commonlit-oof/pred_val105.csv')
    pred_val108 = pd.read_csv('../external/commonlit-oof/pred_val108.csv')

    pred_val105.index = pred_val105['id']
    pred_val105 = pred_val105.reindex(index=pred_val085['id']).reset_index(drop=True)
    pred_val108.index = pred_val108['id']
    pred_val108 = pred_val108.reindex(index=pred_val085['id']).reset_index(drop=True)

    delete_idx = pred_val085[pred_val085.id == '436ce79fe'].index
    pred_val085 = pred_val085.drop(delete_idx).reset_index(drop=True)
    pred_val096 = pred_val096.drop(delete_idx).reset_index(drop=True)
    pred_val105 = pred_val105.drop(delete_idx).reset_index(drop=True)
    pred_val108 = pred_val108.drop(delete_idx).reset_index(drop=True)

    # upura
    pred_val000 = pd.read_csv('../external/commonlit-oof/pred_val000.csv')
    pred_val000.index = pred_val000['id']
    pred_val000 = pred_val000.reindex(index=pred_val085['id']).reset_index(drop=True)

    y_preds = [
        pred_val000.pred_target,
        pred_val085.pred_target,
        pred_val096.pred_target,
        pred_val105.pred_target,
        pred_val108.pred_target,
    ]
    y_true = pred_val085.true_target

    for d in y_preds:
        print(mean_squared_error(y_true, d, squared=False))

    init_state = [round(1 / len(y_preds), 3) for _ in range(len(y_preds) - 1)]
    results = minimize(f, init_state, method='Nelder-Mead')
    print(results)

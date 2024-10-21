import pandas as pd
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.externals.six import StringIO
import pydotplus
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def simple_plot():
    img = io.BytesIO()
    np.random.seed(196801)
    x = np.random.rand(50)
    y = np.random.rand(50)
    plt.plot(x, y)
    plt.savefig(img, format='png')
    img.seek(0)
    '''trace = go.Scatter(
        x=x,
        y=y,
        mode='markers'
    )
    data = [trace]
    graphJson = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)'''
    lp = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return lp


def scatter_plot():
    img = io.BytesIO()

    np.random.seed(19680801)
    x = np.random.rand(30)
    y = np.random.rand(30)
    colors = np.random.rand(30)
    area = np.pi * (15 * np.random.rand(30)) ** 2
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.savefig(img, format='png')
    img.seek(0)

    sp = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return sp


def linear_reg_plot(feature='ENGINESIZE', input='ENGINESIZE'):
    data = pd.read_csv("data/co2_emission.csv")
    lm = LinearRegression()
    img = io.BytesIO()
    X = data[[feature]]
    Y = data['CO2EMISSIONS']
    Y_Label = 'Co2 Emission'
    lm.fit(X, Y)
    if feature == 'MODELYEAR':
        X_Label = 'Model Year'
    elif feature == 'ENGINESIZE':
        X_Label = 'Engine Size'
    elif feature == 'CYLINDERS':
        X_Label = 'Cylinders'
    elif feature == 'FUELCONSUMPTION_CITY':
        X_Label = 'Fuel Consumption in City'
    elif feature == 'FUELCONSUMPTION_HWY':
        X_Label = 'Fuel Consumption in Highway'
    elif feature == 'FUELCONSUMPTION_COMB':
        X_Label = 'Fuel Consumption Comb'
    elif feature == 'FUELCONSUMPTION_COMB_MPG':
        X_Label = 'Fuel Consumption Comb Miles Per Gallon'
    else:
        X_Label = 'Engine Size'
    predict = lm.predict(X)
    intercept = lm.intercept_
    slope = lm.coef_
    plt.plot(X, predict, color='red', label='LR')
    plt.ylabel(Y_Label)
    plt.xlabel(X_Label)
    plt.title('Car\'s Co2 Emission')
    plt.scatter(X, Y, label='Co2 Emission')
    co2_data = go.Scatter(
        x=data[feature].values.tolist(),
        y=data['CO2EMISSIONS'].values.tolist(),
        mode='markers',
        name='Co2 Emission'
    )
    model_prediction = go.Scatter(
        x=data[feature].values.tolist(),
        y=predict,
        name='LR'
    )
    layout = go.Layout(
        title='Co2 Emission Prediction From Cars',
        xaxis=dict(title=X_Label),
        yaxis=dict(title='Co2 Emission')
    )
    chart_data = [co2_data, model_prediction]
    chart = json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)
    chart_layout = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    x = data[feature].values.tolist()
    y = data['CO2EMISSIONS'].values.tolist()
    result = {
        'X': x[:50],
        'Y': y[:50],
        'X_label': X_Label,
        'Predicted': np.around(np.array(predict[:50])),
        'intercept': intercept,
        'slope': slope,
        'msqe': mean_squared_error(Y, predict),
        'r2': r2_score(Y, predict),
        'accuracy': np.floor(r2_score(Y, predict) * 100),
        'chart': plot,
        'plot': chart,
        'plot_layout': chart_layout
    }
    return result


def mlrf_label(feature_1):
    if feature_1 == 'symboling':
        X1_Label = 'Symboling'
    elif feature_1 == 'normalized-losses':
        X1_Label = 'Normalized Loses'
    elif feature_1 == 'num-of-doors':
        X1_Label = 'Number of Doors'
    elif feature_1 == 'drive-wheels':
        X1_Label = 'Drive Wheels'
    elif feature_1 == 'engine-location':
        X1_Label = 'Engine Location (Fwd/Rear)'
    elif feature_1 == 'wheel-base':
        X1_Label = 'Wheel Base'
    elif feature_1 == 'length':
        X1_Label = 'Length'
    elif feature_1 == 'width':
        X1_Label = 'Width'
    elif feature_1 == 'height':
        X1_Label = 'Height'
    elif feature_1 == 'curb-weight':
        X1_Label = 'Curb Weight'
    elif feature_1 == 'num-of-cylinders':
        X1_Label = 'Number of Cylinders'
    elif feature_1 == 'engine-size':
        X1_Label = 'Engine Size'
    elif feature_1 == 'bore':
        X1_Label = 'Bore'
    elif feature_1 == 'stroke':
        X1_Label = 'Stroke'
    elif feature_1 == 'compression-ratio':
        X1_Label = 'Compression Ratio'
    elif feature_1 == 'horsepower':
        X1_Label = 'Horse Power'
    elif feature_1 == 'peak-rpm':
        X1_Label = 'Peak RPM'
    elif feature_1 == 'city-mpg':
        X1_Label = 'City Miles per Gallon'
    elif feature_1 == 'highway-mpg':
        X1_Label = 'Highway Miles Per Gallon'
    elif feature_1 == 'city-L/100km':
        X1_Label = 'City Liters In 100KM'
    elif feature_1 == 'horsepower-binned':
        X1_Label = 'Horse Power Binned'
    elif feature_1 == 'diesel':
        X1_Label = 'Diesel Engine'
    elif feature_1 == 'gas':
        X1_Label = 'Gas Engine'
    else:
        X1_Label = 'Feature'
    return X1_Label


def multiple_linear_reg_plot(feature_1='horsepower', feature_2='engine-size', feature_3='highway-mpg',
                             feature_4='curb-weight'):
    data = pd.read_csv("data/car_prices.csv")
    lm = LinearRegression()
    img = io.BytesIO()
    X = data[[feature_1, feature_2, feature_3, feature_4]]
    Y = data['price']
    lm.fit(X, Y)
    X1_Label = mlrf_label(feature_1)
    X2_Label = mlrf_label(feature_2)
    X3_Label = mlrf_label(feature_3)
    X4_Label = mlrf_label(feature_4)
    predict = lm.predict(X)
    intercept = lm.intercept_
    slope = lm.coef_
    box = dict(facecolor='blue', pad=5, alpha=0.2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.subplots_adjust(left=0.2, wspace=0.6)
    ax1.scatter(Y, data[feature_1], label=X1_Label)
    ax1.plot(predict, data[feature_1], color='red', label='MLR')
    ax1.set_ylabel(X1_Label, bbox=box)
    ax1.set_xlabel("Price", bbox=box)
    ax1.legend()
    ax2.scatter(Y, data[feature_2], label=X2_Label)
    ax2.plot(predict, data[feature_2], color='red', label='MLR')
    ax2.set_ylabel(X2_Label, bbox=box)
    ax2.set_xlabel("Price", bbox=box)
    ax2.legend()
    ax3.scatter(Y, data[feature_3], label=X3_Label)
    ax3.plot(predict, data[feature_3], color='red', label='MLR')
    ax3.set_ylabel(X3_Label, bbox=box)
    ax3.set_xlabel("Price", bbox=box)
    ax3.legend()
    ax4.scatter(Y, data[feature_4], label=X4_Label)
    ax4.plot(predict, data[feature_4], color='red', label='MLR')
    ax4.set_ylabel(X4_Label, bbox=box)
    ax4.set_xlabel("Price", bbox=box)
    ax4.legend()
    plt.title('Car Prices')
    plt.savefig(img, format='png')
    img.seek(0)
    plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    x1 = data[feature_1].values.tolist()
    x2 = data[feature_2].values.tolist()
    x3 = data[feature_3].values.tolist()
    x4 = data[feature_4].values.tolist()
    y = data['price'].values.tolist()

    chart_1_data = go.Scatter(
        x=y,
        y=x1,
        mode='markers',
        name=X1_Label
    )
    chart_1_prediction = go.Scatter(
        x=predict,
        y=x1,
        name='MLR'
    )
    chart_1_lyt = go.Layout(
        title='Car Price Prediction From ' + X1_Label,
        xaxis=dict(title='Price'),
        yaxis=dict(title=X1_Label)
    )

    chart_2_data = go.Scatter(
        x=y,
        y=x2,
        mode='markers',
        name=X2_Label
    )
    chart_2_prediction = go.Scatter(
        x=predict,
        y=x2,
        name='MLR'
    )
    chart_2_lyt = go.Layout(
        title='Car Price Prediction From ' + X2_Label,
        xaxis=dict(title='Price'),
        yaxis=dict(title=X2_Label)
    )

    chart_3_data = go.Scatter(
        x=y,
        y=x3,
        mode='markers',
        name=X3_Label
    )
    chart_3_prediction = go.Scatter(
        x=predict,
        y=x3,
        name='MLR'
    )
    chart_3_lyt = go.Layout(
        title='Car Price Prediction From ' + X3_Label,
        xaxis=dict(title='Price'),
        yaxis=dict(title=X3_Label)
    )

    chart_4_data = go.Scatter(
        x=y,
        y=x4,
        mode='markers',
        name=X4_Label
    )
    chart_4_prediction = go.Scatter(
        x=predict,
        y=x4,
        name='MLR'
    )
    chart_4_lyt = go.Layout(
        title='Car Price Prediction From ' + X4_Label,
        xaxis=dict(title='Price'),
        yaxis=dict(title=X4_Label)
    )
    chart_1_comb = [chart_1_data, chart_1_prediction]
    chart_1 = json.dumps(chart_1_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_1_layout = json.dumps(chart_1_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    chart_2_comb = [chart_2_data, chart_2_prediction]
    chart_2 = json.dumps(chart_2_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_2_layout = json.dumps(chart_2_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    chart_3_comb = [chart_3_data, chart_3_prediction]
    chart_3 = json.dumps(chart_3_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_3_layout = json.dumps(chart_3_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    chart_4_comb = [chart_4_data, chart_4_prediction]
    chart_4 = json.dumps(chart_4_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_4_layout = json.dumps(chart_4_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    result = {
        'X1': x1[:50],
        'X2': x2[:50],
        'X3': x3[:50],
        'X4': x4[:50],
        'Y': y[:50],
        'X1_Label': X1_Label,
        'X2_Label': X2_Label,
        'X3_Label': X3_Label,
        'X4_Label': X4_Label,
        'Predicted': np.around(np.array(predict[:50])),
        'intercept': intercept,
        'slope': slope,
        'msqe': mean_squared_error(Y, predict),
        'r2': r2_score(Y, predict),
        'accuracy': np.floor(r2_score(Y, predict) * 100),
        'chart': plot,
        'chart_1': chart_1,
        'chart_1_layout': chart_1_layout,
        'chart_2': chart_2,
        'chart_2_layout': chart_2_layout,
        'chart_3': chart_3,
        'chart_3_layout': chart_3_layout,
        'chart_4': chart_4,
        'chart_4_layout': chart_4_layout,
    }
    return result


def plr_label(name):
    if name == 'year':
        label = 'Year'
    elif name == 'gdp-usd':
        label = 'GDP Per Year in US$ (Current)'
    elif name == 'military-expenditure-percent-gdp':
        label = 'Military Expenditure of GDP in Percents(%)'
    elif name == 'urban-population':
        label = 'City Population'
    elif name == 'rural-population':
        label = 'Rural Population'
    elif name == 'life-expectancy-at-birth-male-years':
        label = 'Life Expectancy At Birth For Male (In Years)'
    elif name == 'life-expectancy-at-birth-female-years':
        label = 'Life Expectancy At Birth For Female (In Years)'
    elif name == 'life-expectancy-at-birth-total-years':
        label = 'Life Expectancy At Birth In Total (In Years)'
    elif name == 'co2-emissions-kt':
        label = 'Co2 Emission in Kilo Tons Per Year'
    else:
        label = 'Label'
    return label


def polynomial_linear_reg_plot(feature='year', label='gdp-usd', degree=3):
    data = pd.read_csv("data/pakistan_worldbank.csv")
    img = io.BytesIO()
    X = data[[feature]]
    y = data[label]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(xtrain)
    poly_reg.fit(X_poly, ytrain)
    lm = LinearRegression()
    lm.fit(X_poly, ytrain)
    predict = lm.predict(poly_reg.fit_transform(X))
    title = 'Pakistan\'s Data Prediction'
    x_label = plr_label(feature)
    y_label = plr_label(label)

    chart_data = go.Scatter(
        x=data[feature].values.tolist(),
        y=data[label].values.tolist(),
        mode='markers',
        name=y_label
    )
    prediction_chart = go.Scatter(
        x=data[feature].values.tolist(),
        y=predict,
        name='PLR'
    )
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label)
    )
    chart_data = [chart_data, prediction_chart]
    chart = json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)
    chart_layout = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)

    intercept = lm.intercept_
    slope = lm.coef_
    plt.scatter(X, y, color='blue', label=y_label)
    plt.plot(X, lm.predict(poly_reg.fit_transform(X)), color='red', label='POLYR')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    x_ = X.values.tolist()
    y_ = y.values.tolist()
    result = {
        'X': x_[0:],
        'Y': y_[0:],
        'Predicted': np.around(np.array(predict)),
        'intercept': intercept,
        'slope': slope,
        'msqe': mean_squared_error(y, predict),
        'r2': r2_score(y, predict),
        'accuracy': np.floor(r2_score(y, predict) * 100),
        'chart': plot,
        'X_Label': x_label,
        'Y_Label': y_label,
        'chart_data': chart,
        'chart_layout': chart_layout
    }
    return result


def decision_tree_plot(test_size=0.3, label='Drug', depth=4):
    data = pd.read_csv("data/drug200.csv")

    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y = data[label]

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    X[:, 1] = le_sex.transform(X[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=3)
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    drugTree.fit(xtrain, ytrain)
    predict = drugTree.predict(xtest)

    dot_data = StringIO()
    featureNames = data.columns[0:5]
    targetNames = data["Drug"].unique().tolist()

    export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(ytrain), filled=True,
                    special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    plot = base64.b64encode(graph.create_png()).decode()

    xl = xtrain.tolist()

    age = []
    sex = []
    bp = []
    col = []
    nak = []
    for xlv in xl:
        age.append(xlv[0])
        sex.append(xlv[1])
        bp.append(xlv[2])
        col.append(xlv[3])
        nak.append(xlv[4])
    drug = ytrain.tolist()

    xh = xtest.tolist()
    test_age = []
    test_sex = []
    test_bp = []
    test_col = []
    test_nak = []
    for xhv in xh:
        test_age.append(xhv[0])
        test_sex.append(xhv[1])
        test_bp.append(xhv[2])
        test_col.append(xhv[3])
        test_nak.append(xhv[4])

    result = {
        'x1': age,
        'x2': sex,
        'x3': bp,
        'x4': col,
        'x5': nak,
        'y': drug,
        'tx1': test_age,
        'tx2': test_sex,
        'tx3': test_bp,
        'tx4': test_col,
        'tx5': test_nak,
        'Predicted': predict,
        'r2': accuracy_score(ytest, predict),
        'accuracy': np.floor(accuracy_score(ytest, predict) * 100),
        'chart': plot,
    }
    return result


def knn_label(name):
    if name == 'region':
        label = 'Region'
    elif name == 'tenure':
        label = 'Tenure'
    elif name == 'age':
        label = 'Age'
    elif name == 'marital':
        label = 'Marital'
    elif name == 'income':
        label = 'Income'
    elif name == 'ed':
        label = 'Education'
    elif name == 'employ':
        label = 'Employee'
    elif name == 'retire':
        label = 'Retire'
    elif name == 'gender':
        label = 'Gender'
    elif name == 'reside':
        label = 'Reside'
    elif name == 'custcat':
        label = 'Plan'
    else:
        label = 'Label'
    return label


def k_nearest_neighbors_plot(feature_1='age', feature_2='income', k=4):
    data = pd.read_csv("data/tele_cust.csv")
    knn = KNeighborsClassifier(n_neighbors=k)
    img = io.BytesIO()
    X = data[[feature_1, feature_2]]
    # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    Y = data['custcat']
    knn.fit(X, Y)
    X_Label = knn_label(feature_1)
    Y_Label = knn_label(feature_2)

    predict = knn.predict(X)

    basic_service = data[(data['custcat'] == 1)]
    e_service = data[(data['custcat'] == 2)]
    plus_service = data[(data['custcat'] == 3)]
    total_service = data[(data['custcat'] == 4)]

    box = dict(facecolor='blue', pad=5, alpha=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))
    fig.subplots_adjust(left=0.2, wspace=0.6)

    ax1.scatter(basic_service[feature_1][:20], basic_service[feature_2][:20], label='Basic Serice')
    ax1.scatter(e_service[feature_1][:20], e_service[feature_2][:20], label='E-Service')
    ax1.scatter(plus_service[feature_1][:20], plus_service[feature_2][:20], label='Plus Service')
    ax1.scatter(total_service[feature_1][:20], total_service[feature_2][:20], label='Full Service')
    ax1.set_ylabel(X_Label, bbox=box)
    ax1.set_xlabel(Y_Label, bbox=box)
    ax1.legend()

    data['predicted'] = predict
    p_basic_service = data[(data['predicted'] == 1)]
    p_e_service = data[(data['predicted'] == 2)]
    p_plus_service = data[(data['predicted'] == 3)]
    p_total_service = data[(data['predicted'] == 4)]

    ax2.scatter(p_basic_service[feature_1][:20], p_basic_service[feature_2][:20], label='Basic Serice')
    ax2.scatter(p_e_service[feature_1][:20], p_e_service[feature_2][:20], label='E-Service')
    ax2.scatter(p_plus_service[feature_1][:20], p_plus_service[feature_2][:20], label='Plus Service')
    ax2.scatter(p_total_service[feature_1][:20], p_total_service[feature_2][:20], label='Full Service')
    ax2.set_ylabel(X_Label, bbox=box)
    ax2.set_xlabel(Y_Label, bbox=box)
    ax2.legend()

    plt.title('Telecommunication Services')
    plt.savefig(img, format='png')
    img.seek(0)
    plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    x1 = data[feature_1][:20].values.tolist()
    x2 = data[feature_2][:20].values.tolist()
    y = data['custcat'][:20].values.tolist()

    bs_data = go.Scatter(
        x=basic_service[feature_1][:20].values.tolist(),
        y=basic_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Basic Services'
    )
    es_data = go.Scatter(
        x=e_service[feature_1][:20].values.tolist(),
        y=e_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='E-Services'
    )
    ps_data = go.Scatter(
        x=plus_service[feature_1][:20].values.tolist(),
        y=plus_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Plus Services'
    )
    ts_data = go.Scatter(
        x=total_service[feature_1][:20].values.tolist(),
        y=total_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Total Services'
    )
    chart_1_lyt = go.Layout(
        title='Customer Type From ' + X_Label + ' And ' + Y_Label,
        xaxis=dict(title=X_Label),
        yaxis=dict(title=Y_Label)
    )

    pbs_data = go.Scatter(
        x=p_basic_service[feature_1][:20].values.tolist(),
        y=p_basic_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Basic Services'
    )
    pes_data = go.Scatter(
        x=p_e_service[feature_1][:20].values.tolist(),
        y=p_e_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='E-Services'
    )
    pps_data = go.Scatter(
        x=p_plus_service[feature_1][:20].values.tolist(),
        y=p_plus_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Plus Services'
    )
    pts_data = go.Scatter(
        x=p_total_service[feature_1][:20].values.tolist(),
        y=p_total_service[feature_2][:20].values.tolist(),
        mode='markers',
        name='Total Services'
    )
    chart_2_lyt = go.Layout(
        title='Customer Type Prediction From ' + X_Label + ' And ' + Y_Label,
        xaxis=dict(title=X_Label),
        yaxis=dict(title=Y_Label)
    )

    chart_1_comb = [bs_data, es_data, ps_data, ts_data]
    chart_1 = json.dumps(chart_1_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_1_layout = json.dumps(chart_1_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    chart_2_comb = [pbs_data, pes_data, pps_data, pts_data]
    chart_2 = json.dumps(chart_2_comb, cls=plotly.utils.PlotlyJSONEncoder)
    chart_2_layout = json.dumps(chart_2_lyt, cls=plotly.utils.PlotlyJSONEncoder)

    result = {
        'X1': x1[:50],
        'X2': x2[:50],
        'Y': y[:50],
        'X_Label': X_Label,
        'Y_Label': Y_Label,
        'Predicted': np.around(np.array(predict[:50])),
        'msqe': mean_squared_error(Y, predict),
        'r2': accuracy_score(Y, predict),
        'accuracy': np.floor(accuracy_score(Y, predict) * 100),
        'chart': plot,
        'chart_1': chart_1,
        'chart_1_layout': chart_1_layout,
        'chart_2': chart_2,
        'chart_2_layout': chart_2_layout,
    }
    return result


def k_mean_plot(dataset='cust_segmentation.csv', clusters=3, n_int=12):
    dt = pd.read_csv('data/' + dataset)
    data = dt.drop('Address', axis=1)
    img = io.BytesIO()

    X = data.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = preprocessing.StandardScaler().fit_transform(X)

    k_means = KMeans(init="k-means++", n_clusters=clusters, n_init=n_int)
    k_means.fit(X)
    labels = k_means.labels_
    data["Clus_km"] = labels

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')

    ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))

    plt.savefig(img, format='png')
    img.seek(0)
    plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    trace = go.Scatter3d(
        x=X[:, 1],
        y=X[:, 0],
        z=X[:, 3],
        mode='markers',
        marker=dict(
            size=12,
            color=labels,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    data = [trace]
    layout = go.Layout(
        title='Clustering of Telecommunication Customers',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis=dict(
                title='Education'),
            yaxis=dict(
                title='Age'),
            zaxis=dict(
                title='Income'),
        )
    )
    chart = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    chart_layout = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)

    result = {
        'Predicted': labels,
        'chart': plot,
        'plot': chart,
        'x': X[:, 1],
        'y': X[:, 0],
        'z': X[:, 3],
        'chart_layout': chart_layout
    }

    return result

from flask_wtf import Form
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField, FloatField
from wtforms.validators import InputRequired, Email, Length, EqualTo


class SignupForm(Form):
    full_name = StringField('Full Name', [InputRequired("Please enter your full name."),
                                          Length(min=2, max=80,
                                                 message="Full Name should be in between of 2 and 80 characters.")],
                            render_kw={"placeholder": "Full Name"})
    email = StringField('Email', [InputRequired("Please enter your email."), Email("Please enter your email.")],
                        render_kw={"placeholder": "Email"})
    password = PasswordField('Password', [InputRequired("Please enter your password."),
                                          Length(min=6, message="Password should be at least 6 charactors long.")],
                             render_kw={"placeholder": "Password"})
    accept = BooleanField(
        'I agree to the <a href="/terms-and-conditions">terms and conditions</a> and <a href="/privacy-policy">privacy policy</a>.',
        [InputRequired("Please accept our TAC and Privacy Policy.")])
    register = SubmitField('Join')


class SigninForm(Form):
    email = StringField('Email', [InputRequired("Please enter your email."), Email("Please enter your email.")],
                        render_kw={"placeholder": "Email"})
    password = PasswordField('Password', [InputRequired("Please enter your password."),
                                          Length(min=6, message="Password should be at least 6 charactors long.")],
                             render_kw={"placeholder": "Password"})
    login = SubmitField('Login')


class AdminAddNotificationForm(Form):
    title = StringField('Title', [InputRequired("Please enter notificaion title."),
                                  Length(min=2, max=80,
                                         message="title should be in between of 2 and 80 characters.")],
                        render_kw={"placeholder": "Title"})
    description = TextAreaField('description', [Length(max=2000,
                                                       message="Description can be 2000 characters maximum.")],
                                render_kw={"placeholder": "Description"})
    save = SubmitField('Save')


class ContactUsForm(Form):
    full_name = StringField('Your Name', [InputRequired("Please enter your full name."),
                                          Length(min=3, max=80,
                                                 message="Full Name should be in between of 2 and 80 characters.")],
                            render_kw={"placeholder": "Full Name"})
    email = StringField('Your Email', [InputRequired("Please enter your email."), Email("Please enter your email.")],
                        render_kw={"placeholder": "Email"})
    message = TextAreaField('Your Message', [Length(min=7, max=900,
                                                    message="Message should be in between of 7 to 900 characters.")],
                            render_kw={"placeholder": "Message"})
    send = SubmitField('Send')


class ResetPasswordForm(Form):
    email = StringField('Email', [InputRequired("Please enter your email."), Email("Please enter your email.")],
                        render_kw={"placeholder": "Email"})
    reset = SubmitField('Reset Password')


class PasswordResetForm(Form):
    new_password = PasswordField('New Password', [InputRequired("Please enter your new password."),
                                                  Length(min=6,
                                                         message="Password should be at least 6 characters long.")])
    repeat_new_password = PasswordField('Repeat New Password', [InputRequired("Please repeat your new password."),
                                                                Length(min=6,
                                                                       message="Password should be at least 6 characters long."),
                                                                EqualTo('new_password',
                                                                        message="New Password and Repeat New Passwords does not match")])
    reset = SubmitField('Request Password Reset')


class DashboardPasswordReset(Form):
    old_password = PasswordField('Current Password', [InputRequired("Please enter your current password."),
                                                      Length(min=6,
                                                             message="Password should be at least 6 characters long.")],
                                 render_kw={"placeholder": "Current Password"})
    new_password = PasswordField('New Password', [InputRequired("Please enter your new password."),
                                                  Length(min=6,
                                                         message="Password should be at least 6 characters long.")],
                                 render_kw={"placeholder": "New Password"})
    repeat_new_password = PasswordField('Repeat New Password', [InputRequired("Please repeat your new password."),
                                                                Length(min=6,
                                                                       message="Password should be at least 6 characters long."),
                                                                EqualTo('new_password',
                                                                        message="New Password and Repeat New Passwords does not match")],
                                        render_kw={"placeholder": "Repeat New Password"})
    reset = SubmitField('Reset Password')


class DashboardAccountDeactivation(Form):
    password = PasswordField('Password', [InputRequired("Please enter your current password."),
                                          Length(min=6,
                                                 message="Password should be at least 6 characters long.")],
                             render_kw={"placeholder": "Current Password"})
    confirmation = StringField('Please Type "DEACTIVATE" in upper case in the field below.',
                               [InputRequired("Please enter DEACTIVATE below to continue.")],
                               render_kw={"placeholder": "DEACTIVATE"})
    reset = SubmitField('DEACTIVATE ACCOUNT')


class DashboardProfile(Form):
    username = StringField('Username', [InputRequired("Please enter a unique Username."), Length(min=4, max=30,
                                                                                                 message="Username  should be in between of 4 and 30 characters.")],
                           render_kw={"placeholder": "Username"})
    full_name = StringField('Full Name', [InputRequired("Please enter your full name."),
                                          Length(min=2, max=80,
                                                 message="Full Name should be in between of 2 and 80 characters.")],
                            render_kw={"placeholder": "Full Name"})
    first_name = StringField('First Name', [Length(max=45,
                                                   message="First name can be 45 characters maximum.")],
                             render_kw={"placeholder": "First Name"})
    last_name = StringField('Last Name', [Length(max=45,
                                                 message="Last name can be 45 characters maximum.")],
                            render_kw={"placeholder": "Last Name"})
    website = StringField('Website', [Length(max=100,
                                             message="Website URL can be 100 characters maximum.")],
                          render_kw={"placeholder": "Website"})
    country = StringField('Country', [Length(max=45,
                                             message="Country name can be 45 characters maximum.")],
                          render_kw={"placeholder": "Country"})
    github = StringField('Github Username', [Length(max=45,
                                                    message="Github username can be 45 characters maximum.")],
                         render_kw={"placeholder": "Github username"})
    kaggle = StringField('Kaggle Username', [Length(max=45,
                                                    message="Kaggle username can be 45 characters maximum.")],
                         render_kw={"placeholder": "Kaggle username"})
    linkedin = StringField('LinkedIn Username', [Length(max=45,
                                                        message="LinkedIn username can be 45 characters maximum.")],
                           render_kw={"placeholder": "LinkedIn username"})
    bio = TextAreaField('Bio', [Length(max=500,
                                       message="Bio can be 480 characters maximum.")],
                        render_kw={"placeholder": "Bio"})
    save = SubmitField("Save")


class LinearRegressionTrainForm(Form):
    feature = SelectField('Feature', choices=[
        ('ENGINESIZE', 'Engine Size'),
        ('MODELYEAR', 'Model Year'),
        ('CYLINDERS', 'Cylinders'),
        ('FUELCONSUMPTION_CITY', 'Fuel Consumption in City'),
        ('FUELCONSUMPTION_HWY', 'Fuel Consumption in Highway'),
        ('FUELCONSUMPTION_COMB', 'Fuel Consumption Combined'),
        ('FUELCONSUMPTION_COMB_MPG', 'Fuel Consumption Comb Miles Per Gallon')
    ])
    label = SelectField('Label', choices=[
        ('CO2EMISSIONS', 'Co2 Emission')
    ])
    train = SubmitField('Train')


class LinearRegressionPredictForm(Form):
    input = FloatField('Input', [InputRequired("Please enter a value for prediction.")],
                       render_kw={"placeholder": "Input"})
    predict = SubmitField('Predict')


class MultipleLinearRegressionTrainForm(Form):
    feature_1 = SelectField('Feature 1', choices=[
        ('horsepower', 'Horse Power'),
        ('symboling', 'Symboling'),
        ('normalized-losses', 'Normalized Loses'),
        ('num-of-doors', 'Number of Doors'),
        ('drive-wheels', 'Drive Wheels'),
        ('engine-location', 'Engine Location (Fwd/Rear)'),
        ('wheel-base', 'Wheel Base'),
        ('length', 'Length'),
        ('width', 'Width'),
        ('height', 'Height'),
        ('curb-weight', 'Curb Weight'),
        ('num-of-cylinders', 'Number of Cylinders'),
        ('engine-size', 'Engine Size'),
        ('bore', 'Bore'),
        ('stroke', 'Stroke'),
        ('compression-ratio', 'Compression Ratio'),
        ('peak-rpm', 'Peak RPM'),
        ('city-mpg', 'City Miles per Gallon'),
        ('highway-mpg', 'Highway Miles Per Gallon'),
        ('city-L/100km', 'City Liters In 100KM'),
        ('horsepower-binned', 'Horse Power Binned'),
        ('diesel', 'Diesel Engine'),
        ('gas', 'Gas Engine'),
    ])

    feature_2 = SelectField('Feature 2', choices=[
        ('engine-size', 'Engine Size'),
        ('symboling', 'Symboling'),
        ('normalized-losses', 'Normalized Loses'),
        ('num-of-doors', 'Number of Doors'),
        ('drive-wheels', 'Drive Wheels'),
        ('engine-location', 'Engine Location (Fwd/Rear)'),
        ('wheel-base', 'Wheel Base'),
        ('length', 'Length'),
        ('width', 'Width'),
        ('height', 'Height'),
        ('curb-weight', 'Curb Weight'),
        ('num-of-cylinders', 'Number of Cylinders'),
        ('bore', 'Bore'),
        ('stroke', 'Stroke'),
        ('compression-ratio', 'Compression Ratio'),
        ('horsepower', 'Horse Power'),
        ('peak-rpm', 'Peak RPM'),
        ('city-mpg', 'City Miles per Gallon'),
        ('highway-mpg', 'Highway Miles Per Gallon'),
        ('city-L/100km', 'City Liters In 100KM'),
        ('horsepower-binned', 'Horse Power Binned'),
        ('diesel', 'Diesel Engine'),
        ('gas', 'Gas Engine'),
    ])

    feature_3 = SelectField('Feature 3', choices=[
        ('highway-mpg', 'Highway Miles Per Gallon'),
        ('symboling', 'Symboling'),
        ('normalized-losses', 'Normalized Loses'),
        ('num-of-doors', 'Number of Doors'),
        ('drive-wheels', 'Drive Wheels'),
        ('engine-location', 'Engine Location (Fwd/Rear)'),
        ('wheel-base', 'Wheel Base'),
        ('length', 'Length'),
        ('width', 'Width'),
        ('height', 'Height'),
        ('curb-weight', 'Curb Weight'),
        ('num-of-cylinders', 'Number of Cylinders'),
        ('engine-size', 'Engine Size'),
        ('bore', 'Bore'),
        ('stroke', 'Stroke'),
        ('compression-ratio', 'Compression Ratio'),
        ('horsepower', 'Horse Power'),
        ('peak-rpm', 'Peak RPM'),
        ('city-mpg', 'City Miles per Gallon'),
        ('city-L/100km', 'City Liters In 100KM'),
        ('horsepower-binned', 'Horse Power Binned'),
        ('diesel', 'Diesel Engine'),
        ('gas', 'Gas Engine'),
    ])

    feature_4 = SelectField('Feature 4', choices=[
        ('curb-weight', 'Curb Weight'),
        ('symboling', 'Symboling'),
        ('normalized-losses', 'Normalized Loses'),
        ('num-of-doors', 'Number of Doors'),
        ('drive-wheels', 'Drive Wheels'),
        ('engine-location', 'Engine Location (Fwd/Rear)'),
        ('wheel-base', 'Wheel Base'),
        ('length', 'Length'),
        ('width', 'Width'),
        ('height', 'Height'),
        ('num-of-cylinders', 'Number of Cylinders'),
        ('engine-size', 'Engine Size'),
        ('bore', 'Bore'),
        ('stroke', 'Stroke'),
        ('compression-ratio', 'Compression Ratio'),
        ('horsepower', 'Horse Power'),
        ('peak-rpm', 'Peak RPM'),
        ('city-mpg', 'City Miles per Gallon'),
        ('highway-mpg', 'Highway Miles Per Gallon'),
        ('city-L/100km', 'City Liters In 100KM'),
        ('horsepower-binned', 'Horse Power Binned'),
        ('diesel', 'Diesel Engine'),
        ('gas', 'Gas Engine'),
    ])

    label = SelectField('Label', choices=[
        ('price', 'Price')
    ])

    train = SubmitField('Train')


class PolynomialLinearRegressionTrainForm(Form):
    feature = SelectField('Feature', choices=[
        ('year', 'Year'),
        ('gdp-usd', 'GDP Per Year in US$ (Current)'),
        ('military-expenditure-percent-gdp', 'Military Expenditure of GDP in Percents(%)'),
        ('urban-population', 'City Population'),
        ('rural-population', 'Rural Population'),
        ('life-expectancy-at-birth-male-years', 'Life Expectancy At Birth For Male (In Years)'),
        ('life-expectancy-at-birth-female-years', 'Life Expectancy At Birth For Female (In Years)'),
        ('life-expectancy-at-birth-total-years', 'Life Expectancy At Birth In Total (In Years)'),
        ('co2-emissions-kt', 'Co2 Emission in Kilo Tons Per Year')
    ])
    label = SelectField('Label', choices=[
        ('gdp-usd', 'GDP Per Year in US$ (Current)'),
        ('year', 'Year'),
        ('military-expenditure-percent-gdp', 'Military Expenditure of GDP in Percents(%)'),
        ('urban-population', 'City Population'),
        ('rural-population', 'Rural Population'),
        ('life-expectancy-at-birth-male-years', 'Life Expectancy At Birth For Male (In Years)'),
        ('life-expectancy-at-birth-female-years', 'Life Expectancy At Birth For Female (In Years)'),
        ('life-expectancy-at-birth-total-years', 'Life Expectancy At Birth In Total (In Years)'),
        ('co2-emissions-kt', 'Co2 Emission in Kilo Tons Per Year')
    ])
    degree = SelectField('Degree of Polynomial', choices=[
        (3, '3'),
        (1, '1'),
        (2, '2'),
        (4, '4'),
        (5, '5'),
        (6, '6'),
        (7, '7'),
    ], coerce=int)
    train = SubmitField('Train')


class DecisionTreeTrainForm(Form):
    test_size = SelectField('Test Size of Data', choices=[
        (0.3, '30%'),
        (0.1, '10%'),
        (0.2, '20%'),
        (0.4, '40%'),
        (0.5, '50%'),
    ], coerce=float)

    max_depth = SelectField('Maximum Tree Depth', choices=[
        (4, '4'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (5, '5'),
    ], coerce=int)

    label = SelectField('Label', choices=[
        ('Drug', 'Drug')
    ])

    train = SubmitField('Train')


class KNearestNeighborsTrainForm(Form):
    feature_1 = SelectField('Feature 1', choices=[
        ('region', 'Region'),
        ('tenure', 'Tenure'),
        ('age', 'Age'),
        ('marital', 'Martial'),
        ('income', 'Income'),
        ('ed', 'Education'),
        ('employ', 'Employee'),
        ('retire', 'Retire'),
        ('gender', 'Gender'),
    ])

    feature_2 = SelectField('Feature 2', choices=[
        ('age', 'Age'),
        ('region', 'Region'),
        ('tenure', 'Tenure'),
        ('marital', 'Martial'),
        ('income', 'Income'),
        ('ed', 'Education'),
        ('employ', 'Employee'),
        ('retire', 'Retire'),
        ('gender', 'Gender'),
    ])

    k = SelectField('K = ', choices=[
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4'),
        (5, '5'),
        (6, '6'),
        (7, '7'),
        (8, '8'),
        (9, '9'),
    ], coerce=int)

    label = SelectField('Label', choices=[
        ('custcat', 'Plan')
    ])

    train = SubmitField('Train')


class KMeanTrainForm(Form):
    dataset = SelectField('Dataset', choices=[
        ('cust_segmentation.csv', 'Telecommunication Customers')
    ])
    clusters = SelectField('Number of Clusters', choices=[
        (3, '3'),
        (1, '1'),
        (2, '2'),
        (4, '4'),
        (5, '5'),
        (6, '6'),
        (7, '7'),
    ], coerce=int)

    n_init = SelectField('Number of Times K-Mean Will Run Clustering', choices=[
        (12, '12'),
        (7, '7'),
        (8, '8'),
        (9, '9'),
        (10, '10'),
        (11, '11'),
        (13, '13'),
        (14, '14')
    ], coerce=int)

    train = SubmitField('Train')

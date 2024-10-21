from flask import Flask, render_template, redirect, url_for, session, flash, request
from flask_migrate import Migrate
from config import Config
from models import db, User, Notification, ContactUs
from forms import SignupForm, SigninForm, ContactUsForm, ResetPasswordForm, PasswordResetForm, DashboardPasswordReset, \
    DashboardAccountDeactivation, \
    DashboardProfile, LinearRegressionTrainForm, MultipleLinearRegressionTrainForm, \
    PolynomialLinearRegressionTrainForm, DecisionTreeTrainForm, KNearestNeighborsTrainForm, KMeanTrainForm, AdminAddNotificationForm
from flask_mail import Mail, Message
from algorithms import simple_plot, scatter_plot, linear_reg_plot, multiple_linear_reg_plot, polynomial_linear_reg_plot, \
    decision_tree_plot, \
    k_nearest_neighbors_plot, \
    k_mean_plot
from sqlalchemy import desc
import numpy as np
import json
import os

# Add Graphviz to the PATH
os.environ["PATH"] += os.pathsep + 'G:/Program Files (x86)/Graphviz2.38/bin/'

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = Config['SQLALCHEMY_DATABASE_URI']

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config['SQLALCHEMY_TRACK_MODIFICATIONS']

app.config['SECRET_KEY'] = Config['SECRET_KEY']

app.config['MAIL_SENDER'] = Config['MAIL_SENDER']

db.init_app(app)

app.config.update(
    # EMAIL SETTINGS
    MAIL_SENDER=Config['MAIL_SENDER'],
    MAIL_SERVER=Config['MAIL_SERVER'],
    MAIL_PORT=Config['MAIL_PORT'],
    MAIL_USE_SSL=Config['MAIL_USE_SSL'],
    MAIL_USE_TLS=Config['MAIL_USE_TLS'],
    MAIL_USERNAME=Config['MAIL_USERNAME'],
    MAIL_PASSWORD=Config['MAIL_PASSWORD']
)

mail = Mail(app)

migrate = Migrate(app, db)

app.jinja_env.filters['zip'] = zip

''' 
    """""""""""""""""""""""""""
    Restricted Functions      |
    """""""""""""""""""""""""""
'''


def profile_cover():
    return Config['Profile_Covers'][np.random.randint(0, len(Config['Profile_Covers']) - 1)]


def send_email(subject, sender, recipients, html_body):
    msg = Message(subject, sender=sender, recipients=[recipients])
    msg.html = html_body
    mail.send(msg)


def get_user_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR']


def send_registration_email(user):
    token = user.get_token(expires_in=5 * 24 * 3600)
    send_email('[ML Platform] Account Registration',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/registration.html',
                                         user=user, token=token))


def send_password_reset_email(user):
    token = user.get_token()
    send_email('[ML Platform] Reset Your Password',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/reset-password.html',
                                         user=user, token=token))


def send_password_updated_email(user):
    send_email('[ML Platform] Password Updated',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/password-updated.html',
                                         user=user))


def send_contact_us_email(user):
    send_email('[ML Platform] Message Received',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/contact-us.html',
                                         user=user))


def send_deactivation_confirmation_email(user):
    send_email('[ML Platform] Account Deactivate Confirmation',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/deactivate-account.html',
                                         user=user))


def send_admin_deactivation_confirmation_email(user):
    send_email('[ML Platform] Account Deactivated By Admin',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/admin-deactivate-account.html',
                                         user=user))


def send_admin_activation_confirmation_email(user):
    send_email('[ML Platform] Account Activated By Admin',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/admin-activate-account.html',
                                         user=user))


def send_admin_account_verification_email(user):
    send_email('[ML Platform] Account Verified By Admin',
               sender=Config['MAIL_SENDER'],
               recipients=user.email,
               html_body=render_template('emails/admin-verify-account.html',
                                         user=user))


def format_datetime(date_time):
    formated_date_time = date_time.strftime('%A, %d %B %Y At %I:%M:%S %p')
    return formated_date_time


app.jinja_env.filters['datetime'] = format_datetime

''' 
    """"""""""""""""""""""""""""""""""""""""""""""""
    Public / Unauthenticated Pages URL Routes      |
    """"""""""""""""""""""""""""""""""""""""""""""""
'''


@app.route('/')
def home_page():
    page = {
        'title': 'Home',
        'class': 'sidebar-collapse'
    }
    return render_template("public/index.html", page=page)


@app.route('/join', methods=['GET', 'POST'])
def signup_page():
    if session.get('logged_in'):
        return redirect(url_for('dashboard_home_page'))
    else:
        form = SignupForm()
        if form.validate_on_submit():
            email = form.email.data
            user = User.query.filter_by(email=email).first()
            if user is None:
                user = User(form.full_name.data, email, form.password.data)
                db.session.add(user)
                db.session.commit()
                send_registration_email(user)
                flash(u'Registration Successful. Please check your inbox to verify your email address.', 'message')
                return redirect(url_for('signup_page'))
            elif user.verified == 0:
                flash(
                    u'This email belongs to an unverified account. Please verify your email or'
                    u' Contact support if you have any queries.',
                    'error')
                return redirect(url_for('signup_page'))
            elif user.active == 0:
                flash(u'Account Deactivated. Contact support if you have any queries.',
                      'error')
                return redirect(url_for('signup_page'))
            else:
                flash(u'Email Already Exists. If you forgot your password, try resetting it.',
                      'error')
                return redirect(url_for('signup_page'))
        else:
            page = {
                'title': 'Join',
                'class': 'sidebar-collapse',
            }
            return render_template("public/register.html", page=page, form=form)


@app.route('/login', methods=['GET', 'POST'])
def signin_page():
    if session.get('logged_in'):
        return redirect(url_for('dashboard_home_page'))
    else:
        form = SigninForm()
        if form.validate_on_submit():
            email = form.email.data
            password = form.password.data
            user = User.query.filter_by(email=email).first()
            if user is not None and user.check_password(password) and user.active != 0 and user.verified != 0:
                session['full_name'] = user.full_name
                session['id'] = user.id
                session['email'] = email
                session['logged_in'] = True
                return redirect(url_for('dashboard_home_page'))
            elif user is None:
                flash(u'Email does not exists in our system', 'error')
                return redirect(url_for('signin_page'))
            elif user.verified == 0:
                flash(
                    u'You need to verify your email address first. Please verify your email or'
                    u' Contact support if you have any queries.',
                    'error')
                return redirect(url_for('signin_page'))
            elif user.active == 0:
                flash(u'Account Deactivated. Contact support if you have any queries.',
                      'error')
                return redirect(url_for('signin_page'))
            else:
                flash(u'Invalid Email or Password.', 'error')
                return redirect(url_for('signin_page'))
        else:
            page = {
                'title': 'Login',
                'class': 'sidebar-collapse',
            }
            return render_template("public/login.html", page=page, form=form)


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_page():
    if session.get('logged_in'):
        return redirect(url_for('dashboard_home_page'))
    elif session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard_home_page'))
    else:
        form = ResetPasswordForm()
        if form.validate_on_submit():
            email = form.email.data
            user = User.query.filter_by(email=email).first()
            if user is not None and user.active != 0 and user.verified != 0:
                send_password_reset_email(user)
                flash(u'Password Reset Email Sent. Please check your inbox for instructions to reset your password.',
                      'message')
                return redirect(url_for('reset_password_page'))
            elif user.active == 0:
                flash(u'Account Deactivated. Contact support if you have any queries.',
                      'error')
                return redirect(url_for('reset_password_page'))
            elif user.verified == 0:
                flash(
                    u'You need to verify your email address first. Please verify your email or'
                    u' Contact support if you have any queries.',
                    'error')
                return redirect(url_for('reset_password_page'))
            else:
                flash(u'This email does not exists in our system. Try joining us.', 'error')
                return redirect(url_for('reset_password_page'))
        else:
            page = {
                'title': 'Reset Password',
                'class': 'sidebar-collapse'
            }
            return render_template("public/reset-password.html", page=page, form=form)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if session.get('logged_in'):
        return redirect(url_for('dashboard_home_page'))
    elif session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard_home_page'))
    else:
        user = User.verify_token(token)
        if user is None:
            flash(u'Invalid/Expired reset token or User details.', 'error')
            return redirect(url_for('reset_password_page'))
        else:
            form = PasswordResetForm()
            if form.validate_on_submit():
                user.set_password(form.new_password.data)
                db.session.commit()
                flash(u'Your password has been reset.', 'message')
                return redirect(url_for('signin_page'))
        page = {
            'class': 'sidebar-collapse',
            'title': 'Enter New Password',
        }
        return render_template('public/password-reset.html', form=form, page=page, token=token)


@app.route('/verification/<token>')
def verify_email(token):
    if session.get('logged_in'):
        return redirect(url_for('dashboard_home_page'))
    elif session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard_home_page'))
    else:
        user = User.verify_token(token)
        if user is None:
            flash(u'Invalid/Expired reset token or Email address.', 'error')
            return redirect(url_for('signup_page'))
        elif user.verified == 1:
            flash(u'Your account is already verified', 'error')
            return redirect(url_for('signin_page'))
        else:
            user.set_verified()
            db.session.commit()
            flash(u'Your account has been verified.', 'message')
            return redirect(url_for('signin_page'))


@app.route('/user/<username>')
def user_page(username):
    user = User.query.filter_by(username=username, active=1, verified=1).first_or_404()
    page = {
        'title': user.full_name,
        'class': 'profile-page sidebar-collapse',
        'cover': profile_cover()
    }
    return render_template("public/user-page.html", page=page, user=user)


@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    cf = ContactUsForm()
    if session.get('logged_in'):
        cf.full_name.data = session.get('full_name')
        cf.email.data = session.get('email')
    if cf.validate_on_submit():
        contact = ContactUs(
            cf.full_name.data,
            cf.email.data,
            get_user_ip(),
            cf.message.data
        )
        db.session.add(contact)
        db.session.commit()
        send_contact_us_email(contact)
        flash(u'Your message has been sent to us.', 'message')
        return redirect(url_for('contact_page'))
    page = {
        'title': 'Contact',
        'class': 'sidebar-collapse'
    }
    return render_template(
        "public/contact.html",
        page=page,
        cf=cf
    )


@app.route('/about')
def about_page():
    page = {
        'title': 'About',
        'class': 'sidebar-collapse'
    }
    return render_template("public/about.html", page=page)


@app.route('/terms-and-conditions')
def tac_page():
    page = {
        'title': 'Terms and Conditions',
        'class': 'sidebar-collapse'
    }
    return render_template("public/tac.html", page=page)


@app.route('/privacy-policy')
def privacy_page():
    page = {
        'title': 'Privacy Policy',
        'class': 'sidebar-collapse'
    }
    return render_template("public/privacy-policy.html", page=page)


''' 
    """"""""""""""""""""""""""""""""""""""""""""""""
    Sign in Only / authenticated Pages URL Routes  |
    """"""""""""""""""""""""""""""""""""""""""""""""
'''


@app.route('/dashboard')
def dashboard_home_page():
    if session.get('logged_in'):
        data = {
            'simple_plt': simple_plot(),
            'scatter_plt': scatter_plot(),
            'lr': linear_reg_plot(),
            'mlr': multiple_linear_reg_plot(),
            'plr': polynomial_linear_reg_plot(),
            'knn': k_nearest_neighbors_plot(),
            'dt': decision_tree_plot(),
            'kmean': k_mean_plot()
        }
        page = {
            'title': 'Dashboard',
            'class': '',
        }
        return render_template(
            "dashboard/index.html",
            page=page,
            data=data
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform')
def dashboard_platform_page():
    if session.get('logged_in'):
        data = {
            'lr': linear_reg_plot(),
            'mlr': multiple_linear_reg_plot(),
            'plr': polynomial_linear_reg_plot(),
            'dt': decision_tree_plot(),
            'knn': k_nearest_neighbors_plot(),
            'kmean': k_mean_plot()
        }
        page = {
            'title': 'Platform',
            'class': ''
        }
        return render_template("dashboard/platform.html", page=page, data=data)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/linear-regression', methods=['GET', 'POST'])
def linear_reg_page():
    if session.get('logged_in'):
        lrtf = LinearRegressionTrainForm()
        if lrtf.validate_on_submit():
            lr = linear_reg_plot(feature=lrtf.feature.data)
        else:
            lr = linear_reg_plot()
        page = {
            'title': 'Linear Regression',
            'class': '',
        }
        return render_template(
            "dashboard/apps/lr.html",
            page=page,
            lr=lr,
            lrtf=lrtf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/test')
def dashboard_test_page():
    if session.get('logged_in'):
        users = User.query.order_by(User.id).all()
        sp = k_mean_plot()
        page = {
            'title': 'Test Page',
            'class': '',
        }
        return render_template(
            "dashboard/apps/test.html",
            page=page,
            sp=sp,
            users=users
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/multiple-linear-regression', methods=['GET', 'POST'])
def multiple_linear_reg_page():
    if session.get('logged_in'):
        mlrtf = MultipleLinearRegressionTrainForm()
        if mlrtf.validate_on_submit():
            mlr = multiple_linear_reg_plot(
                feature_1=mlrtf.feature_1.data,
                feature_2=mlrtf.feature_2.data,
                feature_3=mlrtf.feature_3.data,
                feature_4=mlrtf.feature_4.data
            )
        else:
            mlr = multiple_linear_reg_plot()
        page = {
            'title': 'Multiple Linear Regression',
            'class': '',
        }
        return render_template(
            "dashboard/apps/mlr.html",
            page=page,
            mlr=mlr,
            mlrtf=mlrtf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/polynomial-linear-regression', methods=['GET', 'POST'])
def polynomial_linear_reg_page():
    if session.get('logged_in'):
        plrtf = PolynomialLinearRegressionTrainForm()
        if plrtf.validate_on_submit():
            # return json.dumps({plrtf.degree.data:type(plrtf.degree.data).__name__})
            plr = polynomial_linear_reg_plot(
                feature=plrtf.feature.data,
                label=plrtf.label.data,
                degree=plrtf.degree.data
            )
        else:
            plr = polynomial_linear_reg_plot()
        page = {
            'title': 'Polynomial Linear Regression',
            'class': '',
        }
        return render_template(
            "dashboard/apps/plr.html",
            page=page,
            plr=plr,
            plrtf=plrtf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/decision-trees', methods=['GET', 'POST'])
def decision_trees_page():
    if session.get('logged_in'):
        dttf = DecisionTreeTrainForm()
        if dttf.validate_on_submit():
            dt = decision_tree_plot(dttf.test_size.data, dttf.label.data, dttf.max_depth.data)
        else:
            dt = decision_tree_plot()
        page = {
            'title': 'Decision Trees',
            'class': '',
        }
        return render_template(
            "dashboard/apps/dt.html",
            page=page,
            dt=dt,
            dttf=dttf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/k-nearest-neighbors', methods=['GET', 'POST'])
def knn_page():
    if session.get('logged_in'):
        knntf = KNearestNeighborsTrainForm()
        if knntf.validate_on_submit():
            knn = k_nearest_neighbors_plot(
                feature_1=knntf.feature_1.data,
                feature_2=knntf.feature_2.data,
                k=knntf.k.data
            )
        else:
            knn = k_nearest_neighbors_plot()
        page = {
            'title': 'K Nearest Neighbors',
            'class': '',
        }
        return render_template(
            "dashboard/apps/knn.html",
            page=page,
            knn=knn,
            knntf=knntf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/platform/k-mean', methods=['GET', 'POST'])
def kmean_page():
    if session.get('logged_in'):
        kmtf = KMeanTrainForm()
        if kmtf.validate_on_submit():
            kmean = k_mean_plot(
                dataset=kmtf.dataset.data,
                clusters=kmtf.clusters.data,
                n_int=kmtf.n_init.data
            )
        else:
            kmean = k_mean_plot()
        page = {
            'title': 'K-Mean',
            'class': '',
        }
        return render_template(
            "dashboard/apps/kmean.html",
            page=page,
            kmean=kmean,
            kmtf=kmtf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/notifications')
def dashboard_notification_page():
    if session.get('logged_in'):
        notifications = Notification.query.filter_by(active=1).order_by(desc(Notification.id)).limit(10).all()
        page = {
            'title': 'Notifications',
            'class': ''
        }
        return render_template(
            "dashboard/notifications.html",
            page=page,
            notifications=notifications
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/profile')
def dashboard_profile_page():
    if session.get('logged_in'):

        uid = session.get('id')
        user = User.query.get(uid)

        # User Profile Form
        profile_form = DashboardProfile()
        # Password Reset form
        reset_pass_form = DashboardPasswordReset()
        # Account Deletion form
        account_deactivate_form = DashboardAccountDeactivation()

        page = {
            'title': 'Profile',
            'class': '',
            'cover': profile_cover()
        }
        return render_template("dashboard/profile.html",
                               page=page,
                               user=user,
                               profile_form=profile_form,
                               reset_pass_form=reset_pass_form,
                               account_deactivate_form=account_deactivate_form)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/profile/save', methods=['GET', 'POST'])
def update_user_profile():
    if session.get('logged_in'):
        if request.method == 'POST':
            uid = session.get('id')
            user = User.query.get(uid)

            # User Profile Form
            profile_form = DashboardProfile()
            # Password Reset form
            reset_pass_form = DashboardPasswordReset()
            # Account Deletion form
            account_deactivate_form = DashboardAccountDeactivation()

            # Password Reset Code
            if profile_form.validate_on_submit():
                username = profile_form.username.data.lower()
                full_name = profile_form.full_name.data
                first_name = profile_form.first_name.data
                last_name = profile_form.last_name.data
                website = profile_form.website.data
                country = profile_form.country.data
                github = profile_form.github.data
                kaggle = profile_form.kaggle.data
                linkedin = profile_form.linkedin.data
                bio = profile_form.bio.data
                username_check = User.query.filter_by(username=username).first()
                if username_check is None or username_check.id == uid:
                    user.update_full_user(
                        username=username,
                        full_name=full_name,
                        first_name=first_name,
                        last_name=last_name,
                        website=website,
                        country=country,
                        github=github,
                        kaggle=kaggle,
                        linkedin=linkedin,
                        bio=bio
                    )
                    db.session.commit()
                    flash(u'Profile Updated Successfully.', 'pr_message')
                else:
                    flash(u'Username already taken. Please select another username', 'pr_error')

            page = {
                'title': 'Profile',
                'class': '',
                'cover': profile_cover()
            }
            return render_template("dashboard/profile.html",
                                   page=page,
                                   user=user,
                                   profile_form=profile_form,
                                   reset_pass_form=reset_pass_form,
                                   account_deactivate_form=account_deactivate_form)
        else:
            return redirect(url_for('dashboard_profile_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/profile/reset-password', methods=['GET', 'POST'])
def update_profile_password():
    if session.get('logged_in'):
        if request.method == 'POST':
            uid = session.get('id')
            user = User.query.get(uid)

            # User Profile Form
            profile_form = DashboardProfile()
            # Password Reset form
            reset_pass_form = DashboardPasswordReset()
            # Account Deletion form
            account_deactivate_form = DashboardAccountDeactivation()

            # Password Reset Code
            if reset_pass_form.validate_on_submit():
                old_password = reset_pass_form.old_password.data
                new_password = reset_pass_form.new_password.data
                if old_password == new_password:
                    flash(u'Current Password and New Password are same.', 'rp_error')
                elif user.check_password(old_password):
                    user.set_password(new_password)
                    db.session.commit()
                    flash(u'Password Reset Successful.', 'rp_message')
                    send_password_updated_email(user)
                else:
                    flash(u'Old Password does not match with Current Password.', 'rp_error')

            page = {
                'title': 'Profile',
                'class': '',
                'cover': profile_cover()
            }
            return render_template("dashboard/profile.html",
                                   page=page,
                                   user=user,
                                   profile_form=profile_form,
                                   reset_pass_form=reset_pass_form,
                                   account_deactivate_form=account_deactivate_form)
        else:
            return redirect(url_for('dashboard_profile_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/profile/deactivate', methods=['GET', 'POST'])
def deactivate_account():
    if session.get('logged_in'):
        if request.method == 'POST':
            uid = session.get('id')
            user = User.query.get(uid)

            # User Profile Form
            profile_form = DashboardProfile()
            # Password Reset form
            reset_pass_form = DashboardPasswordReset()
            # Account Deletion form
            account_deactivate_form = DashboardAccountDeactivation()
            if account_deactivate_form.validate_on_submit():
                password = account_deactivate_form.password.data
                confirmation = account_deactivate_form.confirmation.data
                if user.check_password(password):
                    if confirmation != "DEACTIVATE":
                        flash(u'Please type DEACTIVATE in the confirmation field.', 'ad_error')
                    else:
                        user.set_non_active()
                        db.session.commit()
                        send_deactivation_confirmation_email(user)
                        return redirect(url_for('dashboard_logout_page'))
                else:
                    flash(u'Entered Password does not match with Current Password.', 'ad_error')
            page = {
                'title': 'Profile',
                'class': '',
                'cover': profile_cover()
            }
            return render_template("dashboard/profile.html",
                                   page=page,
                                   user=user,
                                   profile_form=profile_form,
                                   reset_pass_form=reset_pass_form,
                                   account_deactivate_form=account_deactivate_form)
        else:
            return redirect(url_for('dashboard_profile_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


@app.route('/dashboard/logout')
def dashboard_logout_page():
    if session.get('logged_in'):
        session.pop('email')
        session.pop('logged_in')
        flash(u'Logout Successful.', 'message')
        return redirect(url_for('signin_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('signin_page'))


''' 
    """"""""""""""""""""""""""""""""""""""""""""""""
    Admin Only / Admin Dashboard Pages URL Routes  |
    """"""""""""""""""""""""""""""""""""""""""""""""
'''


@app.route('/admin/')
def admin_home_page():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard_home_page'))
    else:
        return redirect(url_for('admin_login_page'))


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login_page():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard_home_page'))
    else:
        form = SigninForm()
        if form.validate_on_submit():
            email = form.email.data
            password = form.password.data
            user = User.query.filter_by(email=email, id=1).first()
            if user is not None and user.check_password(password):
                session['full_name'] = user.full_name
                session['id'] = user.id
                session['email'] = email
                session['admin_logged_in'] = True
                return redirect(url_for('admin_dashboard_home_page'))
            else:
                flash(u'Invalid Email or Password.', 'error')
                return redirect(url_for('admin_login_page'))
        else:
            page = {
                'title': 'Admin Login',
                'class': 'sidebar-collapse',
            }
            return render_template("admin/login.html", page=page, form=form)


@app.route('/admin/dashboard')
def admin_dashboard_home_page():
    if session.get('admin_logged_in'):
        total_users = User.query.order_by(User.id).all()
        active_users = User.query.filter_by(active=1, verified=1).all()
        deactivated_users = User.query.filter_by(active=0).all()
        unverified_users = User.query.filter_by(verified=0).all()

        latest_users = User.query.order_by(desc(User.id)).limit(5).all()
        latest_messages = ContactUs.query.order_by(desc(ContactUs.id)).limit(5).all()

        if total_users is None:
            total_users = []
        if active_users is None:
            active_users = []
        if deactivated_users is None:
            deactivated_users = []
        if unverified_users is None:
            unverified_users = []

        analytics = {
            'total_users': total_users,
            'active_users': active_users,
            'deactivated_users': deactivated_users,
            'unverified_users': unverified_users,
            'latest_users': latest_users,
            'latest_messages': latest_messages
        }

        page = {
            'title': 'Admin Dashboard',
            'class': '',
        }
        return render_template(
            "admin/index.html",
            page=page,
            analytics=analytics
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users')
def admin_users_page():
    if session.get('admin_logged_in'):
        users = User.query.order_by(desc(User.id)).all()
        page = {
            'title': 'Users',
            'class': '',
        }
        return render_template(
            "admin/users.html",
            page=page,
            users=users
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/delete/<uid>', methods=['GET', 'POST'])
def admin_delete_user_page(uid):
    if session.get('admin_logged_in'):
        uid = int(uid)
        if uid == 1:
            flash(u'Admin User Cannot be Deleted(User With ID: 1 is Admin).', 'error')
            return redirect(url_for('admin_users_page'))
        else:
            user = User.query.get(uid)
            user_email = user.email
            db.session.delete(user)
            db.session.commit()
            flash(u'User: ' + user_email + ' Deleted Successfully.', 'message')
            return redirect(url_for('admin_users_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/edit/<uid>', methods=['GET', 'POST'])
def admin_edit_user_page(uid):
    if session.get('admin_logged_in'):
        page = {
            'title': 'Admin Dashboard',
            'class': '',
        }
        return render_template("admin/index.html", page=page)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/deactivate/<uid>', methods=['GET', 'POST'])
def admin_deactivate_user_page(uid):
    if session.get('admin_logged_in'):
        uid = int(uid)
        if uid == 1:
            flash(u'Admin User Cannot be Deactivated(User With ID: 1 is Admin).', 'error')
            return redirect(url_for('admin_users_page'))
        else:
            user = User.query.get(uid)
            user_email = user.email
            if user.active == 0:
                flash(u'User: ' + user_email + ' is Already Deactivated.', 'error')
                return redirect(url_for('admin_users_page'))
            else:
                user.set_non_active()
                db.session.commit()
                send_admin_deactivation_confirmation_email(user)
                db.session.commit()
                flash(u'User: ' + user_email + ' Deactivated Successfully.', 'message')
                return redirect(url_for('admin_users_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/activate/<uid>', methods=['GET', 'POST'])
def admin_activate_user_page(uid):
    if session.get('admin_logged_in'):
        uid = int(uid)
        user = User.query.get(uid)
        user_email = user.email
        if user.active == 1:
            flash(u'User: ' + user_email + ' is Already Active.', 'error')
            return redirect(url_for('admin_users_page'))
        else:
            user.set_active()
            db.session.commit()
            send_admin_activation_confirmation_email(user)
            db.session.commit()
            flash(u'User: ' + user_email + ' Activated Successfully.', 'message')
            return redirect(url_for('admin_users_page'))

    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/verify/<uid>', methods=['GET', 'POST'])
def admin_verify_user_page(uid):
    if session.get('admin_logged_in'):
        uid = int(uid)
        user = User.query.get(uid)
        user_email = user.email
        if user.verified == 1:
            flash(u'User: ' + user_email + ' is Already Verified.', 'error')
            return redirect(url_for('admin_users_page'))
        else:
            user.set_verified()
            db.session.commit()
            send_admin_account_verification_email(user)
            flash(u'User: ' + user_email + ' Verified Successfully.', 'message')
            return redirect(url_for('admin_users_page'))


@app.route('/admin/users/verified')
def admin_verified_users_page():
    if session.get('admin_logged_in'):
        users = User.query.filter_by(verified=1).order_by(desc(User.id)).all()
        page = {
            'title': 'Verified Users',
            'class': '',
        }
        return render_template("admin/users.html", page=page, users=users)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/unverified')
def admin_unverified_users_page():
    if session.get('admin_logged_in'):
        users = User.query.filter_by(verified=0).order_by(desc(User.id)).all()
        page = {
            'title': 'Unverified Users',
            'class': '',
        }
        return render_template("admin/unverified-users.html", page=page, users=users)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/active')
def admin_active_users_page():
    if session.get('admin_logged_in'):
        users = User.query.filter_by(active=1).order_by(desc(User.id)).all()
        page = {
            'title': 'Active Users',
            'class': '',
        }
        return render_template("admin/index.html", page=page, users=users)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/users/deactivated')
def admin_deactivated_users_page():
    if session.get('admin_logged_in'):
        users = User.query.filter_by(active=0).order_by(desc(User.id)).all()
        page = {
            'title': 'Deactivated Users',
            'class': '',
        }
        return render_template("admin/deactivated-users.html", page=page, users=users)
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/messages')
def admin_messages_page():
    if session.get('admin_logged_in'):
        messages = ContactUs.query.order_by(desc(ContactUs.id)).all()
        page = {
            'title': 'User Messages',
            'class': '',
        }
        return render_template(
            "admin/messages.html",
            page=page,
            messages=messages
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/messages/<mid>/responded')
def admin_responded_messages_page(mid):
    if session.get('admin_logged_in'):
        mid = int(mid)
        message = ContactUs.query.get(mid)
        message.mark_responded()
        db.session.commit()
        flash(u'Message From ' + message.email + ' marked as responded.', 'message')
        return redirect(url_for('admin_messages_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/messages/<mid>/delete')
def admin_delete_messages_page(mid):
    if session.get('admin_logged_in'):
        mid = int(mid)
        message = ContactUs.query.get(mid)
        user_email = message.email
        db.session.delete(message)
        db.session.commit()
        flash(u'Message From ' + user_email + ' Deleted Successfully.', 'message')
        return redirect(url_for('admin_messages_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/notifications')
def admin_notifications_page():
    if session.get('admin_logged_in'):
        notifications = Notification.query.order_by(desc(Notification.id)).all()
        page = {
            'title': 'Notifications',
            'class': '',
        }
        return render_template(
            "admin/notifications.html",
            page=page,
            notifications=notifications
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/notifications/add', methods=['GET', 'POST'])
def admin_add_notification_page():
    if session.get('admin_logged_in'):
        nf = AdminAddNotificationForm()
        if nf.validate_on_submit():
            notification = Notification(
                nf.title.data,
                nf.description.data,
                session.get('id')
            )
            db.session.add(notification)
            db.session.commit()
            flash(u'Notification Added Successfully.', 'message')
            return redirect(url_for('admin_add_notification_page'))
        page = {
            'title': 'Add Notification',
            'class': '',
        }
        return render_template(
            "admin/add-notification.html",
            page=page,
            nf=nf
        )
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/notifications/not-active/<nid>')
def admin_nonactive_notification_page(nid):
    if session.get('admin_logged_in'):
        notification = Notification.query.get(nid)
        notification.set_inactive()
        db.session.commit()
        flash(u'Notification marked as inactive.', 'message')
        return redirect(url_for('admin_notifications_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/notifications/delete/<nid>')
def admin_delete_notification_page(nid):
    if session.get('admin_logged_in'):
        notification = Notification.query.get(nid)
        db.session.delete(notification)
        db.session.commit()
        flash(u'Notification deleted successfully.', 'message')
        return redirect(url_for('admin_notifications_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


@app.route('/admin/logout')
def admin_logout_page():
    if session.get('admin_logged_in'):
        session.pop('email')
        session.pop('admin_logged_in')
        flash(u'Logout Successful.', 'message')
        return redirect(url_for('admin_login_page'))
    else:
        flash(u'You Need To Login First.', 'error')
        return redirect(url_for('admin_login_page'))


if __name__ == '__main__':
    app.run()

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from time import time
from datetime import datetime
from hashlib import md5
from config import Config
import jwt

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(60), unique=True)
    username = db.Column(db.String(60))
    hashed_password = db.Column(db.String(150))
    website = db.Column(db.String(90))
    country = db.Column(db.String(50))
    bio = db.Column(db.String(1000))
    github = db.Column(db.String(50))
    kaggle = db.Column(db.String(50))
    linkedin = db.Column(db.String(50))
    verified = db.Column(db.SMALLINT)
    active = db.Column(db.SMALLINT)
    time_created = db.Column(db.DateTime)
    time_updated = db.Column(db.DateTime)

    def __init__(self, full_name, email, password):
        self.full_name = full_name
        self.email = email.lower()
        self.username = email.lower()
        self.set_password(password)
        self.active = 1
        self.verified = 0
        dt = datetime.now()
        self.time_created = dt
        self.time_updated = dt

    def update_full_user(self, username, full_name, first_name, last_name, website, country, github, kaggle, linkedin,
                         bio):
        self.username = username
        self.full_name = full_name
        self.first_name = first_name
        self.last_name = last_name
        self.website = website
        self.country = country
        self.github = github
        self.kaggle = kaggle
        self.linkedin = linkedin
        self.bio = bio
        self.set_time_updated()

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)
        self.time_updated = datetime.now()

    def set_time_updated(self):
        self.time_updated = datetime.now()

    def set_non_active(self):
        self.active = 0
        self.time_updated = datetime.now()

    def set_active(self):
        self.active = 1
        self.time_updated = datetime.now()

    def set_verified(self):
        self.verified = 1
        self.time_updated = datetime.now()

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)

    def get_token(self, expires_in=600):
        data = {
            'reset-password': self.email,
            'expires': time() + expires_in
        }
        algorithm = 'HS256'
        return jwt.encode(data, Config['SECRET_KEY'], algorithm=algorithm).decode('utf-8')

    @staticmethod
    def verify_token(token):
        algorithm = 'HS256'
        try:
            email = jwt.decode(token, Config['SECRET_KEY'], algorithm=algorithm)['reset-password']
        except:
            return
        return User.query.filter_by(email=email).first()

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(digest, size)


class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    short_desc = db.Column(db.String(100))
    long_desc = db.Column(db.String(5000))
    notification_by = db.Column(db.Integer)
    active = db.Column(db.SMALLINT)
    time_created = db.Column(db.DateTime)
    time_updated = db.Column(db.DateTime)

    def __init__(self, short_desc, long_desc, notif_by):
        self.short_desc = short_desc
        self.long_desc = long_desc
        self.notification_by = notif_by
        self.active = 1
        dt = datetime.now()
        self.time_created = dt
        self.time_updated = dt

    def set_inactive(self):
        self.active = 0
        self.time_updated = datetime.now()


class ContactUs(db.Model):
    __tablename__ = 'contact_us'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(60))
    ip = db.Column(db.String(50))
    message = db.Column(db.String(1000))
    responded = db.Column(db.SMALLINT)
    time_created = db.Column(db.DateTime)
    time_updated = db.Column(db.DateTime)

    def __init__(self, name, email, ip, message):
        self.name = name
        self.email = email
        self.ip = ip
        self.message = message
        self.responded = 0
        dt = datetime.now()
        self.time_created = dt
        self.time_updated = dt

    def mark_responded(self):
        self.responded = 1
        self.time_updated = datetime.now()

import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))

Config = {}

Config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

Config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

Config['SECRET_KEY'] =  os.getenv('SECRET_KEY')
Config['MAIL_SENDER'] = os.getenv('MAIL_SENDER')
Config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
Config['MAIL_PORT'] = os.getenv('MAIL_PORT')
Config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL')
Config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS')
Config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
Config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

Config['Profile_Covers'] = [
    'cover-ai-music.jpg',
    'cover-ai-waiter.jpg',
    'cover-beach-top.jpg',
    'cover-electric.jpg',
    'cover-fog-mountains.jpg',
    'cover-mountains.jpg',
    'cover-road.jpg',
    'cover-tiger.jpg',
    'cover-underwater.jpg'
]

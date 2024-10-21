# Machine Learning Platform

---

## Introduction

**Machine Learning Platform** was my final year project for my BS Computer Science degree.

ML Platform is a machine learning platform which showcases diƦerent
machine learning algorithm models and allows the users to change the
input to the model to see how the output of the model changes based on
input data as well as the model’s accuracy and error output. There is also
an admin panel available to manage the users and ban / unban the users
based on any of the feedback.

## Installation

### Prerequisites

You need to have the following installed on your system:

- Python 3.6
- pip
- virtualenv
- PostgreSQL 9.6 or any other database server ( optional, as you can default back to sqlite )
- Graphviz ( for generating decision tree images )
- SMTP server ( for sending emails )

### Steps

1. Clone the repository

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/zstarpak/machine-learning-platform.git
```

2. Create a virtual environment

Create a virtual environment using the following command:

```bash
cd machine-learning-platform
python3 -m venv venv
```

3. Activate the virtual environment

Activate the virtual environment using the following command:

```bash
source venv/bin/activate
```

4. Install the dependencies

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file

You can copy the `.env.example` file to create a `.env` file and set the environment variables in it.

6. Run the migrations

We are using the flask-migrate extension to manage the database migrations. Run the following commands to create the database and run the migrations:

```bash
flask db init
flask db migrate
flask db upgrade
```

7. Run the application

Run the application using the following command:

```bash
flask run
```

## ~~Deployment to heroku ( Old )~~

~~heroku buildbox of graphviz is required for this app.~~
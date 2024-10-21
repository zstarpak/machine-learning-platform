"""contact_us table

Revision ID: de31531a7061
Revises: dcfb7558705b
Create Date: 2018-07-31 23:55:36.510025

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'de31531a7061'
down_revision = 'dcfb7558705b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('contact_us',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=True),
    sa.Column('email', sa.String(length=60), nullable=True),
    sa.Column('ip', sa.String(length=50), nullable=True),
    sa.Column('message', sa.String(length=1000), nullable=True),
    sa.Column('responded', sa.SMALLINT(), nullable=True),
    sa.Column('time_created', sa.DateTime(), nullable=True),
    sa.Column('time_updated', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('contact_us')
    # ### end Alembic commands ###
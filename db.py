from peewee import *
from datetime import datetime

db = SqliteDatabase('position.db')

class Position(Model):
    Time = DateTimeField(default=datetime.now)
    Latitude = FloatField()
    Longitude = FloatField()
    class Meta:
        database = db # This model uses the "people.db" database.


db.create_tables([Position])
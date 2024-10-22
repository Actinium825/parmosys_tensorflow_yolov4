import re
from appwrite.exception import AppwriteException
from core.config import cfg
from core import utils
from data.env import Env
from absl.flags import FLAGS

cache = dict()
attribute_key = 'availability'


def init_cache(database):
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    for _, value in class_names.items():
        spot_number = get_spot_number(value)

        if cache.get(spot_number) is None:
            cache[spot_number] = True

    if FLAGS.database == 'appwrite':
        try:
            database.create_boolean_attribute(
                database_id=Env.database_id,
                collection_id=FLAGS.area,
                key=attribute_key,
                required=False,
                default=True,
            )
        except AppwriteException:
            pass

    elif FLAGS.database == 'firebase':
        database.collection(FLAGS.area).get()


def update_database(found_classes, database):
    for found_class in found_classes:
        is_open = found_class[0] == 'O'
        spot_number = get_spot_number(found_class)
        cached_availability = cache.get(spot_number)

        if cached_availability != is_open:
            cache[spot_number] = is_open
            document_id = f'spot{spot_number}'

            if FLAGS.database == 'appwrite':
                try:
                    database.update_document(
                        database_id=Env.database_id,
                        collection_id=FLAGS.area,
                        document_id=document_id,
                        data={attribute_key: is_open},
                    )
                except AppwriteException:
                    database.create_document(
                        database_id=Env.database_id,
                        collection_id=FLAGS.area,
                        document_id=document_id,
                        data={attribute_key: is_open},
                    )

            elif FLAGS.database == 'firebase':
                database.collection(FLAGS.area).document(document_id).set({
                    '$id': document_id,
                    '$collectionId': FLAGS.area,
                    attribute_key: is_open,
                })

def get_spot_number(string: str):
    spot_number = re.findall(r'\d+', string)

    if len(spot_number) == 0:
        spot_number = 1
    else:
        spot_number = spot_number[0]

    return spot_number

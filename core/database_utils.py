import re
from appwrite.exception import AppwriteException
from core.config import cfg
from core import utils
from absl.flags import FLAGS

cache = dict()
attribute_key = 'availability'
appwrite_database_id = 'parking_spaces'


def init_cache(database):
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    for _, value in class_names.items():
        spot_number = get_spot_number(value)

        if cache.get(spot_number) is None:
            cache[spot_number] = True

    if FLAGS.database == 'firebase':
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
                        database_id=appwrite_database_id,
                        collection_id=FLAGS.area,
                        document_id=document_id,
                        data={attribute_key: is_open},
                    )
                except AppwriteException as e:
                    if e.type == 'database_not_found':
                        database.create(
                            database_id=appwrite_database_id,
                            name='Parking Spaces',
                        )
                        create_appwrite_collection(database, document_id, is_open)
                    elif e.type == 'collection_not_found':
                        create_appwrite_collection(database, document_id, is_open)
                    else:
                        database.create_document(
                            database_id=appwrite_database_id,
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


def create_appwrite_collection(database, document_id, is_open):
    database.create_collection(
        database_id=appwrite_database_id,
        collection_id=FLAGS.area,
        name=FLAGS.area,
        permissions=["read(\"guests\")"],
    )

    database.create_boolean_attribute(
        database_id=appwrite_database_id,
        collection_id=FLAGS.area,
        key=attribute_key,
        required=False,
        default=True,
    )

    database.create_document(
        database_id=appwrite_database_id,
        collection_id=FLAGS.area,
        document_id=document_id,
        data={attribute_key: is_open},
    )

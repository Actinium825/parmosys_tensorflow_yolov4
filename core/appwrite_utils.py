import re
import threading
from appwrite.exception import AppwriteException
from core.config import cfg
from core import utils
from data.env import Env
from absl.flags import FLAGS
from appwrite.client import Client
from appwrite.services.databases import Databases

client = Client()
(client
 .set_endpoint(Env.endpoint)
 .set_project(Env.project_id)
 .set_key(Env.api_key)
 )
database = Databases(client)
cache = dict()
attribute_key = 'attribute'


def init_cache():
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    for _, value in class_names.items():
        spot_number = re.findall(r'\d+', value)[0]

        if cache.get(spot_number) is None:
            cache[spot_number] = True

    try:
        database.create_boolean_attribute(
            database_id=Env.database_id,
            collection_id=FLAGS.update,
            key=attribute_key,
            required=False,
            default=True,
        )
    except AppwriteException:
        pass


def update_database(found_classes):
    functions = []
    threads = []

    for found_class in found_classes:
        is_open = found_class[0] == 'O'
        spot_number = re.findall(r'\d+', found_class)[0]
        cached_availability = cache.get(spot_number)

        if cached_availability != is_open:
            cache[spot_number] = is_open

            def update():
                try:
                    database.update_document(
                        database_id=Env.database_id,
                        collection_id=FLAGS.update,
                        document_id=f'spot{spot_number}',
                        data={attribute_key: is_open},
                    )

                except AppwriteException:
                    database.create_document(
                        database_id=Env.database_id,
                        collection_id=FLAGS.update,
                        document_id=f'spot{spot_number}',
                        data={attribute_key: is_open},
                    )

            functions.append(update)

    for function in functions:
        thread = threading.Thread(target=function)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

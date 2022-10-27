key = b'\xc5\x9a\x16\x91\x18\xc7\x9f\x82\xe3R\xe0\x1b~\xc3\x07\x14'
host="facepay-pg-dev.postgres.database.azure.com"
port="5432"
user="admin_user"
password="F@cep@y#123"
database="facepay"
schema_name = 'ford_face_data_dev'
# schema_name = 'ford_face_data_qa'
register_svp_table_name = 'product_users'
register_svp_audit_table_name = 'product_users_audit'
register_table_name = 'face_pay_registration'
register_image_table_name = 'registration_image'
recognize_table_name = 'face_pay_recognition'

ip_address_api = '52.183.128.252'
port_api = '60000'
update_status_url = f'http://{ip_address_api}:{port_api}/update_user_status'
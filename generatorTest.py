import ketos.data_handling.database_interface as dbi

from ketos.data_handling.data_feeding import BatchGenerator




############################################################################


h5filename = 'db_h5__file'


# open h5d database for reading
db = dbi.open_file(h5filename, 'r')
print(db)
train_data = dbi.open_table(db, "/train/table_data")
train_annot = dbi.open_table(db,"/train/table_annot")
val_data = dbi.open_table(db, "/eval/table_data")
val_annot = dbi.open_table(db,"/eval/table_annot")
test_data = dbi.open_table(db, "/test/table_data")
test_annot = dbi.open_table(db, "/test/table_annot")



train_generator = BatchGenerator(batch_size=128, data_table=train_data, annot_table=train_annot, annot_in_data_table=False,shuffle=True, refresh_on_epoch_end=True)

val_generator = BatchGenerator(batch_size=32, data_table=val_data, annot_in_data_table=True,
                                 shuffle=True, refresh_on_epoch_end=False)



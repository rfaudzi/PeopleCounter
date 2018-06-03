import pymysql
import datetime

class RekapPengunjung:

    def __init__(self, host, username, password, db):
        self.host = host
        self.username = username
        self.password = password
        self.db = db
        self.conn = pymysql.connect(host=self.host, user=self.username, password=self.password, db=self.db)

    def save(self):
        db = self.conn.cursor()
        now = datetime.datetime.now()
        tgl = now.strftime("%Y-%m-%d")
        jam = 'pukul'+now.strftime("%H")

        query = """SELECT * FROM RekapPengunjung WHERE tanggal = "%s" """ %\
                (tgl)

        data = db.execute(query)

        if(data == 0):
            # insert data awal
            query = """INSERT INTO RekapPengunjung \
                    VALUES ("%s", %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)""" % \
                    (tgl, 0,0,0,0,0,0,0,0,0,0,0,0)
            try:
                db.execute(query)
                self.conn.commit()

                # update data pengunjung
                query = """UPDATE RekapPengunjung SET %s = %s + 1 WHERE tanggal = "%s" """ % \
                        (jam, jam, tgl)

                try:
                    db.execute(query)
                    self.conn.commit()
                except:
                    self.conn.rollback()
            except:
                self.conn.rollback()
        else:
            #update data pengunjung
            query = """UPDATE RekapPengunjung SET %s = %s + 1 WHERE tanggal = "%s" """ % \
                    (jam, jam, tgl)

            try:
                db.execute(query)
                self.conn.commit()
            except:
                self.conn.rollback()

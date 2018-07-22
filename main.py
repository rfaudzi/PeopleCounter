import people_counter_v2
import rekap_pengunjung

class Main:

    def main(self):
        host = 'localhost'
        username = 'root'
        password = ''
        db = 'rekap_pengunjung'
        PeopleCounter = people_counter_v2.PeopleCounter(host, username, password, db)
        # PeopleCounter = people_counter.PeopleCounter(host, username, password, db)
        PeopleCounter.counter()
        
run = Main()
run.main()

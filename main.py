import people_counter
import rekap_pengunjung

class Main:

    def main(self):
        host = 'localhost'
        username = 'sabda'
        password = 'root'
        db = 'people_counter'
        PeopleCounter = people_counter.PeopleCounter(host, username, password, db)
        PeopleCounter.counter()
        
run = Main()
run.main()

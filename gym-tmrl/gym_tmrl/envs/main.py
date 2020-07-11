# import library

from gym_tmrl.envs.read_game import screen_record

if __name__ == '__main__':
    print("Wait 2 sec")
    #time.sleep(2)
    print("Go")
    screen_record(tool=["fps","vis","get_speed", "road", "radar"])
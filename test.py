from inputs import get_mouse
while 1:
     events = get_mouse()
     for event in events:
         print(event.ev_type, event.code, event.state)

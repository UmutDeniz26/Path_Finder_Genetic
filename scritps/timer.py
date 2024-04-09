import time

class Timer:
    def __init__(self):
        self.timers = []
    
    def start_new_timer(self, timer_name):
        self.update_timers()
        for timer in self.timers:
            if timer["name"] == timer_name:
                timer["start"] = time.time()
                timer["end"] = time.time()
                timer["total_duration"] += timer["summ"]
                timer["update_cnt"] += 1
                return "updated"
        self.timers.append(
                {
                    "name":timer_name,
                    "start": time.time(),
                    "end": -1,
                    "total_duration":0.00001,
                    "update_cnt":0
                }
            )
        return "appended"
    
    def stop_timer(self, timer_name):
        for timer in self.timers:
            if timer["name"] == timer_name:
                timer["end"] = time.time()
                return "stopped"
        return "timer not found"
    
    def remove_timer(self, timer_name):
        for timer in self.timers:
            if timer["name"] == timer_name:
                self.timers.remove(timer)
                return "removed"
        return "timer not found"
    
    def reset_timer(self, timer_name):
        for timer in self.timers:
            if timer["name"] == timer_name:
                timer["start"] = time.time()
                timer["end"] = -1
                timer["total_duration"] = 0.00001
                timer["update_cnt"] = 0
                return "reset"
        return "timer not found"

    def print_ratio(self, timer_name, timer_name2):
        
        for timer in self.timers:
            if timer["name"] == timer_name:
                timer1 = timer
            if timer["name"] == timer_name2:
                timer2 = timer
        
        self.update_timers()
        print()
        try:
            if timer1 and timer2:
                time1 = timer1["total_duration"] if timer1["total_duration"] > 1 else timer1["summ"]
                time2 = timer2["total_duration"] if timer2["total_duration"] > 1 else timer2["summ"]

                if time1 > time2:
                    print(f"{timer_name2} is %{(time2/time1)*100:>14.10f} of {timer_name}")
                    print(f"({timer_name} / {timer_name2} = {time1:7.4f} / {time2:7.4f} = {time1/time2:>14.10f})")
                    
                else:
                    print(f"{timer_name} is %{(time1/time2)*100:>14.10f} of {timer_name2}")
                    print(f"({timer_name2} / {timer_name} = {time2:7.4f} / {time1:7.4f} = {time2/time1:>14.10f})")
                    
        except:
            print(f"Error occured while calculating the ratio of {timer_name} and {timer_name2}")  
        print()
        return "timer not found"

    def update_timers(self):
        for timer in self.timers:
            current_time = (time.time() - timer["start"]) if timer["end"] == -1 else 0
            end_between = (timer["end"] - timer["start"]) if timer["end"] != -1 else 0
            total_duration = timer["total_duration"]
            average = total_duration / (timer["update_cnt"] if timer["update_cnt"] != 0 else 1)
            
            timer.update({"current": current_time})
            timer.update({"end_between": end_between})
            timer.update({"average": average})
            timer.update({"summ": current_time + end_between })

    def get_timer_index(self, timer_name):
        for index, timer in enumerate(self.timers):
            if timer["name"] == timer_name:
                return index
        return -1

    def print_timers(self):
        print("All timers are returned in seconds")
        print("Names                       Duration( current ) - Duration( end ) - Average - Update Count - Total Duration")
        self.update_timers()
        for timer in self.timers:
            print(f"{timer['name']:<30}   {timer['current']:>14.10f}  {timer['end_between']:>14.10f}  {timer['average']:>14.10f} {timer['update_cnt']:>14} {timer['update_cnt']*timer['average']:>14.10f}")
            
def main():
    timer = Timer()
    timer.start_new_timer("timer1")
    time.sleep(1)
    timer.start_new_timer("timer1")
    time.sleep(1)
    timer.start_new_timer("timer2")
    time.sleep(1)
    timer.stop_timer("timer1")
    timer.start_new_timer("timer3")
    time.sleep(2)
    timer.print_timers()

if __name__ == "__main__":
    main()
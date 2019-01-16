import time

prev_err = None
sum_err = 0
prev_time = None

offset = 0
K_P = 0
K_I = 0
K_D = 0

def calculate_angle(err):
    global prev_err, sum_err, prev_time

    if prev_time == None:
        prev_time = time.time()
        prev_err = err
        return

    cur_time = time.time()
    dt = cur_time - prev_time

    derr = (err - prev_err) / dt
    sum_err = sum_err + err

    angle = offset + K_P * err + K_D * derr + K_I * sum_err
    return angle

def main():
    pass

if __name__ == '__main__':
    main()

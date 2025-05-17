import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
TRIG = 16
ECHO = 18

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
print('Waiting a few seconds for the sensor to settle')
time.sleep(2)

try:
    while True:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        timeout = time.time() + 0.04  # 40ms timeout (~6.8m max)

        while GPIO.input(ECHO) == 0 and time.time() < timeout:
            pulse_start = time.time()

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == 1 and time.time() < timeout:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        if pulse_duration < 0.04:
            distance = pulse_duration * 17165
            distance = round(distance, 1)
            print('Distance:', distance, 'cm')
        else:
            print("âŒ Mesure Ã©chouÃ©e (pas d'objet dÃ©tectÃ©)")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nArrÃªt du programme.")
    GPIO.cleanup()

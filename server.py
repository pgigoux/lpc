#!/usr/bin/env python
"""
LPC Simulator

- Commands
CFGLGS       Configure the active LGS Units
CALCPOINT    LGS pointing offsets calculation
GETPOINTOFF  Get calculated LGS pointing offsets
SETBACKTHR   Set Background Threshold
GETDIAG'     Get Diagnostics
LPCSHTDWN'   Request to Shutdown
STANDBY'     Request to STANDBY
ONLINE       Request to ONLINE

- Error codes and messages
1, 'Wrong command'
2, 'Wrong number of parameters, expected <n>'
3: 'Wrong syntax, parameter <n>'
4, 'LGS positions not found'
5, 'Error taking image'
6, 'Error loading image'
7, 'Not enough stars detected'
8, 'Astrometry failed'
9, 'Photometry failed'
10, 'Error saving FITS'
11, 'High scattering background'
12, 'Communication failure'
13, 'Target temperature not reached'
14, 'Failed to go ONLINE',
15, 'Failed to go STANDBY'
16, 'Disk full'
17, 'LPC cover not opened'
18, 'Failed to set active LGS'
19, 'Failed to set background threshold'
20, 'Failed to shutdown'
21, 'LPC cover not closed'
22, 'Rotator motor not initialized'
23, 'Cover motor not initialized'
24, 'Astrometry/photometry timeout'
25, 'Laser ON time longer than exposure time'
26, 'Target temperature out of range (0-30'

- Alarms
LGS FAILURE              0x1
COVER NOT OPENED         0x2
COMMUNICATION FAILURE    0x4
TEMPERATURE NOT REACHED  0x8
COVER NOT CLOSED         0x10
ONLINE FAILED            0x20
STANDBY FAILED           0x40
ROTATOR NOT INITIALIZED  0x80
COVER NOT INITIALIZED    0x100
HIGH SCATTER             0x10000
DISK FULL                0x20000
"""
import argparse
import socket
import random
from enum import Enum, auto
from threading import Thread

# LPC commands
# Configure the active LGS Units
CMD_CONFIGURE_LGS = 'CFGLGS'
# LGS pointing offsets calculation
CMD_CALCULATE_POINTING = 'CALCPOINT'
# Get calculated LGS pointing offsets
CMD_GET_LGS_OFFSETS = 'GETPOINTOFF'
# Set Background Threshold
CMD_SET_BACKGROUND = 'SETBACKTHR'
# Get Diagnostics
CMD_GET_DIAGNOSTICS = 'GETDIAG'
# Request to Shutdown
CMD_SHUTDOWN = 'LPCSHTDWN'
# Request to STANDBY
CMD_STANDBY = 'STANDBY'
# Request to ONLINE
CMD_ONLINE = 'ONLINE'
# Dump LPC state (debugging)
CMD_DUMP = 'dump'

# Dictionary with command names and expected number of arguments
command_dict = {
    CMD_CONFIGURE_LGS: 8,
    CMD_CALCULATE_POINTING: 17,
    CMD_GET_LGS_OFFSETS: 0,
    CMD_SET_BACKGROUND: 1,
    CMD_GET_DIAGNOSTICS: 0,
    CMD_SHUTDOWN: 0,
    CMD_STANDBY: 0,
    CMD_ONLINE: 1,
    CMD_DUMP: 0
}

# Error codes
OK = 0
ERROR_BAD_COMMAND = 1
ERROR_NUMBER_OF_ARGUMENTS = 2
ERROR_PARAMETER_SYNTAX = 3
ERROR_TEMPERATURE = 26

# Error messages
error_messages = {
    OK: 'Ok',
    ERROR_BAD_COMMAND: 'Wrong command',
    ERROR_NUMBER_OF_ARGUMENTS: 'Wrong number of parameters, expected',
    ERROR_PARAMETER_SYNTAX: 'Wrong syntax, parameter',
    ERROR_TEMPERATURE: 'Target temperature out of range (0-30)'
}

# Temperature range
TEMP_MIN = 10
TEMP_MAX = 30

# Value used for disabled LGS
UNDEFINED_VALUE = -999


# LPC states
class State(Enum):
    ONLINE = auto()
    STANDBY = auto()


class LPC:
    """
    Class used to keep the LPC state
    """
    current_state = State.STANDBY
    alarm_flag = 0
    calc_running_flag = False
    percent_complete = 0  # %
    data_ready_flag = False
    busy_flag = False
    rotator_angle = 40.8
    peltier_control_flag = False
    peltier_temp = 10  # 0-30
    peltier_duty_cycle = 10  # %
    ready_to_expose_flag = False
    argon_refill = 0  # not used
    simulation_flag = False
    store_image_flag = False
    photometry_flag = False
    square_size = 10  # arc seconds
    background_threshold = 2300  # adu
    lgs_active = [False, False, False, False]
    laser_on_time = 2  # tenths of second
    exposure_time = 8
    beam_height = 11  # meters
    simulation_file_name = 'simulation_file'

    @classmethod
    def get_lgs_active(cls) -> str:
        """
        Get active LGSs
        :return: string of the form 'T,F,T,T'
        """
        return ','.join(str(bool_to_str(_)) for _ in cls.lgs_active)

    @classmethod
    def get_configuration(cls) -> str:
        """
        This function is called when the GETDIAG command is received
        :return: command response
        """
        return f'{state_to_string(cls.current_state)},' \
               f'{cls.alarm_flag},' \
               f'{bool_to_str(cls.calc_running_flag)},' \
               f'{cls.percent_complete},' \
               f'{bool_to_str(cls.data_ready_flag)},' \
               f'{bool_to_str(cls.busy_flag)},' \
               f'{cls.rotator_angle},' \
               f'{bool_to_str(cls.peltier_control_flag)},' \
               f'{cls.peltier_temp},' \
               f'{cls.peltier_duty_cycle},' \
               f'{bool_to_str(cls.ready_to_expose_flag)},' \
               f'{cls.argon_refill},' \
               f'{bool_to_str(cls.simulation_flag)},' \
               f'{bool_to_str(cls.store_image_flag)},' \
               f'{bool_to_str(cls.photometry_flag)},' \
               f'{cls.square_size},' \
               f'{cls.background_threshold},' \
               f'{cls.get_lgs_active()},' \
               f'{cls.laser_on_time},' \
               f'{cls.exposure_time},' \
               f'{cls.beam_height:.1f},' \
               f'{cls.simulation_file_name}'

    @classmethod
    def dump(cls) -> str:
        """
        Print the LPC internal variables to the standard output
        Used for debugging
        :return:
        """
        print('-' * 70)
        print(f'State                 {cls.current_state}')
        print(f'Alarms                {cls.alarm_flag}')
        print(f'Running flag          {cls.calc_running_flag}')
        print(f'Percent complete      {cls.percent_complete}')
        print(f'Data ready flag       {cls.data_ready_flag}')
        print(f'Busy flag             {cls.busy_flag}')
        print(f'Rotator angle         {cls.rotator_angle}')
        print(f'Peltier control       {cls.peltier_control_flag}')
        print(f'Peltier temperature   {cls.peltier_temp}')
        print(f'Peltier duty cycle    {cls.peltier_duty_cycle}')
        print(f'Ready to exposure     {cls.ready_to_expose_flag}')
        print(f'Argon refill          {cls.argon_refill}')
        print(f'Simulation fl;ag      {cls.simulation_flag}')
        print(f'Store image flag      {cls.store_image_flag}')
        print(f'Photometry flag       {cls.photometry_flag}')
        print(f'Square size           {cls.square_size}')
        print(f'Background threshold  {cls.background_threshold}')
        print(f'LGS active flags      {cls.lgs_active}')
        print(f'Laser on time         {cls.laser_on_time}')
        print(f'Exposure time         {cls.exposure_time}')
        print(f'Beam height           {cls.beam_height}')
        print('-' * 70)
        return error(OK)

    @classmethod
    def configure_lgs(cls, lgs_active: list, laser_on_time: int,
                      lpc_integration_time: int, simulation_flag: bool) -> str:
        """
        This function is called when the CFGLGS is received
        :param lgs_active: list of active LGS
        :param laser_on_time: ime the laser is on [seconds]
        :param lpc_integration_time: integration time [seconds]
        :param simulation_flag: put the LPC in simulation mode?
        :return: command response (OK)
        """
        cls.lgs_active = lgs_active.copy()
        cls.laser_on_time = laser_on_time
        cls.exposure_time = lpc_integration_time
        cls.simulation_flag = simulation_flag
        return error(OK)

    @classmethod
    def calculate_pointing(cls, ra: float, dec: float, angle: float, alt: float, az: float, utc: str,
                           alt1: float, az1: float, alt2: float, az2: float,
                           alt3: float, az3: float, alt4: float, az4: float,
                           store: bool, photometry: bool, size: float):
        """
        This function is called when the CALCPOINT command is received.
        :param ra: telescope RA [deg]
        :param dec: telescope DEC [deg]
        :param angle: parallactic angle [deg]
        :param alt: telescope altitude [deg]
        :param az: telescope azimuth [deg]
        :param utc: UTC (e.g. 2015-03-09T14:57:24.703)
        :param alt1: lgs21 alt coordinate relative to telescope axis [arcsec]
        :param az1: lgs1 az coordi2nate relative to telescope axis [arcsec]
        :param alt2: same for lgs1
        :param az2: same for lgs2
        :param alt3: same for lgs3
        :param az3: same for lgs3
        :param alt4: same for lgs4
        :param az4: same for lgs4
        :param store: store image?
        :param photometry: request Photometry calculation?
        :param size: maximum LGS distance allowed from launch telescope coordinates [arcsec]
        :return: command response (OK)
        """
        LPC.store_image_flag = store
        LPC.photometry_flag = photometry
        LPC.square_size = size
        return error(OK)

    @classmethod
    def get_point_offsets(cls) -> str:
        """
        This function is called when the GETPOINTOFF is received
        :return: command response
        """
        file_name = 'image.fits' if LPC.store_image_flag else 'NO_FILE'

        alt_off1, az_off1 = random_pair(10, 20) if LPC.lgs_active[0] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        alt_off2, az_off2 = random_pair(10, 20) if LPC.lgs_active[1] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        alt_off3, az_off3 = random_pair(10, 20) if LPC.lgs_active[2] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        alt_off4, az_off4 = random_pair(20, 20) if LPC.lgs_active[3] else (UNDEFINED_VALUE, UNDEFINED_VALUE)

        cent_x, cent_y = random_pair(240, 250)

        x_lgs1, y_lgs1 = random_pair(200, 300) if LPC.lgs_active[0] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        x_lgs2, y_lgs2 = random_pair(200, 300) if LPC.lgs_active[1] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        x_lgs3, y_lgs3 = random_pair(200, 300) if LPC.lgs_active[2] else (UNDEFINED_VALUE, UNDEFINED_VALUE)
        x_lgs4, y_lgs4 = random_pair(200, 300) if LPC.lgs_active[3] else (UNDEFINED_VALUE, UNDEFINED_VALUE)

        flux_1 = random.uniform(10, 1000) if LPC.lgs_active[0] else UNDEFINED_VALUE
        flux_2 = random.uniform(10, 1000) if LPC.lgs_active[1] else UNDEFINED_VALUE
        flux_3 = random.uniform(10, 1000) if LPC.lgs_active[2] else UNDEFINED_VALUE
        flux_4 = random.uniform(10, 1000) if LPC.lgs_active[3] else UNDEFINED_VALUE

        fwhm_1 = random.uniform(1, 10) if LPC.lgs_active[0] else UNDEFINED_VALUE
        fwhm_2 = random.uniform(1, 10) if LPC.lgs_active[1] else UNDEFINED_VALUE
        fwhm_3 = random.uniform(1, 10) if LPC.lgs_active[2] else UNDEFINED_VALUE
        fwhm_4 = random.uniform(1, 10) if LPC.lgs_active[3] else UNDEFINED_VALUE

        scat_1 = random.uniform(10, 20) if LPC.lgs_active[0] else UNDEFINED_VALUE
        scat_2 = random.uniform(10, 20) if LPC.lgs_active[1] else UNDEFINED_VALUE
        scat_3 = random.uniform(10, 20) if LPC.lgs_active[2] else UNDEFINED_VALUE
        scat_4 = random.uniform(10, 20) if LPC.lgs_active[3] else UNDEFINED_VALUE

        background = random.uniform(10, 2000)

        error_code = 11  # high scattering background

        # print('off', alt_off1, az_off1, alt_off2, az_off2, alt_off3, az_off3, alt_off4, az_off4)
        # print('cent', cent_x, cent_y)
        # print('x,y', x_lgs1, y_lgs1, x_lgs2, y_lgs2, x_lgs3, y_lgs3, x_lgs4, y_lgs4)
        # print('fwhm', fwhm_1, fwhm_2, fwhm_3, fwhm_4)
        # print('flux', flux_1, flux_2, flux_3, flux_4)
        # print('scat', scat_1, scat_2, scat_3, scat_4)
        # print('back', background)
        return f'{file_name},' \
               f'{alt_off1:.1f},{az_off1:.1f},{alt_off2:.1f},{az_off2:.1f},' \
               f'{alt_off3:.1f},{az_off3:.1f},{alt_off4:.1f},{az_off4:.1f},' \
               f'{cent_x:.2f},{cent_y:.2f},' \
               f'{x_lgs1:.2f},{y_lgs1:.2f},{x_lgs2:.2f},{y_lgs2:.2f},' \
               f'{x_lgs3:.2f},{y_lgs3:.2f},{x_lgs4:.2f},{y_lgs4:.2f},' \
               f'{fwhm_1},{fwhm_2},{fwhm_3},{fwhm_4}' \
               f'{flux_1},{flux_2},{flux_3},{flux_4},' \
               f'{scat_1},{scat_2}{scat_3},{scat_4},' \
               f'{background},' \
               f'{error_code}'

    @classmethod
    def set_background(cls, value: float):
        """
        This function is called when the SETBACKTHR is received
        :param value: background threshold [adu]
        :return: command response (OK)
        """
        LPC.background_threshold = value
        return error(OK)

    @classmethod
    def online(cls, temperature: float):
        """
        This function is called when the ONLINE command is received
        :param temperature: requested sensor operating temperature [C]
        :return: command response (OK or ERROR_TEMPERATURE)
        """
        if TEMP_MIN <= temperature <= TEMP_MAX:
            cls.peltier_temp = temperature
            cls.current_state = State.ONLINE
            return error(OK)
        else:
            return error(ERROR_TEMPERATURE)

    @classmethod
    def standby(cls):
        """
        This function is called when the STANDBY command is received
        :return: command response (OK)
        """
        cls.current_state = State.STANDBY
        return error(OK)


def state_to_string(state: State) -> str:
    """
    Convert the state enumeration into a string
    :param state: State.ONLINE or State.STANDBY
    :return: 'ONLINE' or 'STANDBY'
    """
    return 'ONLINE' if state == State.ONLINE else 'STANDBY'


def random_pair(min_val: float, max_val: float) -> tuple:
    """
    Return a tuple with two random float numbers
    :param min_val: min value
    :param max_val:  max value
    :return: tuple with random numbers
    """
    return random.uniform(min_val, max_val), random.uniform(min_val, max_val)


def bool_to_str(flag: bool) -> str:
    """
    Convert boolean into a string
    :param flag: boolean value
    :return: 'T' or 'F'
    """
    return 'T' if flag else 'F'


def str_to_bool(flag: str):
    """
    Convert a 'T' or 'F' string into a boolean value
    :param flag: 'T' or 'F'
    :return: True or False
    """
    return True if flag == 'T' else False


def error(error_code: int, value=0):
    """
    Format the error response
    :param error_code: error code
    :param value: optional value used in some errors
    :return: error string
    """
    msg = error_messages[error_code]
    if error_code == OK:
        return f'{msg}'
    elif error_code in [ERROR_NUMBER_OF_ARGUMENTS, ERROR_PARAMETER_SYNTAX]:
        return f'Error {error_code} {msg} {value}'
    else:
        return f'Error {error_code} {msg}'


def command_configure_lgs(args: list) -> str:
    """
    Check the CFGLGS arguments and execute the command
    :param args: command arguments
    :return: error string
    """
    try:
        lgs_active = [str_to_bool(x) for x in args[0:4]]
        lgs_on = int(args[4])
        exp_time = int(args[5])
        sim_flag = str_to_bool(args[6])
        return LPC.configure_lgs(lgs_active, lgs_on, exp_time, sim_flag)
    except ValueError:
        error(ERROR_PARAMETER_SYNTAX, 0)


def calculate_pointing(args: list) -> str:
    """
    Check the CALCPOINT arguments and execute the command
    :param args: command arguments
    :return: error string
    """
    try:
        ra, dec, angle, alt, az = float(args[0]), float(args[1]), float(args[2]), float(args[3]), float(args[4])
        utc = args[5]
        alt1, az1, alt2, az2, alt3, az3, alt4, az4 = float(args[6]), float(args[7]), float(args[8]), \
                                                     float(args[9]), float(args[10]), float(args[11]), \
                                                     float(args[12]), float(args[13]),
        store = str_to_bool(args[14])
        photometry = str_to_bool(args[15])
        size = float(args[16])
        return LPC.calculate_pointing(ra, dec, angle, alt, az, utc, alt1, az1, alt2, az2,
                                      alt3, az3, alt4, az4, store, photometry, size)
    except ValueError:
        return error(ERROR_PARAMETER_SYNTAX, 0)


def command_set_background(args: list):
    """
    Check the SETBACKTHR arguments and execute the command
    :param args: command arguments
    :return: error string
    """
    try:
        background = float(args[0])
    except ValueError:
        return error(ERROR_PARAMETER_SYNTAX, 1)
    return LPC.set_background(background)


def command_online(args: list):
    """
    Check the ONLINE arguments and execute the command
    :param args: command arguments
    :return: error string
    """
    try:
        temperature = float(args[0])
    except ValueError:
        return error(ERROR_PARAMETER_SYNTAX, 1)
    return LPC.online(temperature)


def process_command(command: str) -> str:
    """
    Process command received from the client
    Split the command arguments into a list
    :param command: command
    :return: error message to send back to the client
    """
    print('process_command', command)

    # Get command and arguments
    command = command.replace(',', ' ').split()
    cmd = command[0]
    args = command[1:]
    n_args = len(args)

    # Check number of arguments
    if n_args != command_dict[cmd]:
        return error(ERROR_NUMBER_OF_ARGUMENTS, n_args)

    # Now process the command
    if cmd == CMD_CONFIGURE_LGS:
        return command_configure_lgs(args)
    elif cmd == CMD_CALCULATE_POINTING:
        return calculate_pointing(args)
    elif cmd == CMD_GET_LGS_OFFSETS:
        return LPC.get_point_offsets()
    elif cmd == CMD_SET_BACKGROUND:
        return command_set_background(args)
    elif cmd == CMD_GET_DIAGNOSTICS:
        return LPC.get_configuration()
    elif cmd == CMD_SHUTDOWN:
        return error(OK)
    elif cmd == CMD_ONLINE:
        return command_online(args)
    elif cmd == CMD_STANDBY:
        return LPC.standby()
    elif cmd == CMD_DUMP:
        return LPC.dump()
    else:
        return error(ERROR_BAD_COMMAND)


def client_thread(connection: socket.socket):
    """
    Function in charge of receiving commands from a client
    :param connection: socket connection
    """
    while True:
        data = connection.recv(2048)
        if not data:
            print('client disconnected')
            break
        data = data.decode('utf-8').strip()
        if len(data) > 0:
            print('Received', data)
            response = process_command(data)
            connection.sendall(response.encode('utf-8'))
    connection.close()


def start_server(host: str, port: int):
    """
    Start the server that listen for client connection
    More than one client can connect, each connection will be handled by a separate client thread.
    :param host: host ip address
    :param port: port number
    :return:
    """
    server_socket = socket.socket()
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        print(str(e))

    print('waiting for a connection..')
    server_socket.listen(5)

    try:
        while True:
            client, address = server_socket.accept()
            print('connected to: ' + address[0] + ':' + str(address[1]))
            th = Thread(target=client_thread, args=(client,))
            th.start()
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        print('closing server socket')
        server_socket.close()


def test():
    """
    Test all commands
    """
    print(process_command(CMD_STANDBY))
    print(process_command(CMD_GET_DIAGNOSTICS))
    print(process_command(f'{CMD_ONLINE} 15'))
    print(process_command(CMD_GET_DIAGNOSTICS))
    print(process_command(f'{CMD_CONFIGURE_LGS} T,T,F,T,2,11,0,F'))
    print(process_command(
        f'{CMD_CALCULATE_POINTING}'
        f' 25.5,72.4,20,30.5,175.5,2015-03-09T14:57:24.703,30.7,175.3,30.7,170.7,30.3,175.3,30.3,170.7,T,F,10'))
    print(process_command(CMD_GET_LGS_OFFSETS))
    print(process_command(f'{CMD_SET_BACKGROUND} 1200'))
    print(process_command(CMD_SHUTDOWN))
    print(process_command(CMD_DUMP))


if __name__ == '__main__':
    # Process command line options
    parser = argparse.ArgumentParser('LPC simulator')

    parser.add_argument('host',
                        action='store',
                        default='',
                        help='host name or ip')

    parser.add_argument('port',
                        action='store',
                        type=int,
                        default=7777,
                        help='port number')

    c_args = parser.parse_args()

    # Start the server
    start_server(c_args.host, c_args.port)

#include <stdint.h>
#include <math.h>
#include "config.h"
#include "modes.h"
#include "robot.h"
#include "module.h"
#include "registers.h"
#include "hardware.h"
#include "lutmath.h"

#ifdef HAS_SD
#include "conf/config.h"
#include "efs.h"
#include "ls.h"
#include "mkfs.h"
#include <stdio.h>
#endif

#ifdef HAS_CAN
#include "can.h"
#endif

/*
#define FLAG_NULL		0
#define FLAG_READ_ON	0
#define FLAG_READ_OFF	1
*/
#define EPS M_PI/6.0

// Two oscillators per body element
#define OSC_COUNT 2*SEG_COUNT

#ifdef HAS_SD
EmbeddedFileSystem	sd_efs;
EmbeddedFile		sd_file_position;
EmbeddedFile		sd_file_setpoint;
char				sd_buff [128];
int8_t				pos_buff [4375];
uint32_t            time_buff [625];
#endif

#ifdef HAS_LOG
#define LOG_SIZE 1024
int8_t set_buff[LOG_SIZE * SEG_COUNT];
int8_t pos_buff[LOG_SIZE * SEG_COUNT];
uint8_t delta_t_buff[LOG_SIZE];
#define log_pos reg16_table[REG16_LOG_POS]
#endif

static uint8_t flag_can;
static int8_t positions[SEG_COUNT]={0};

// Device table.  List of device drivers for newlib.
const struct device_table_entry *device_table[] = {0};

void modes_mb_callback(uint8_t write, uint16_t addr, uint8_t* data)
{
  uint8_t i;
  if (!write)
  {
    switch (addr)
    {
      case REGMB_POS:
        data[0] = SEG_COUNT;
        for (i = 0; i < SEG_COUNT; i++)
        {
          data[i+1] = positions[i];
        }
        break;
#ifdef HAS_LOG
#if (SEG_COUNT > 14)
#error Too many segments declared, data could not fit in multibyte register.
#endif
      case REGMB_LOG_ENTRY: {
        uint8_t n = 0;
        uint16_t pos = log_pos * SEG_COUNT;
        data[0] = 2 * SEG_COUNT + 1;
        data[++n] = delta_t_buff[log_pos];
        for (i = 0; i < SEG_COUNT; i++) {
          data[++n] = set_buff[pos + i];
        }
        for (i = 0; i < SEG_COUNT; i++) {
          data[++n] = pos_buff[pos + i];
        }
        log_pos++;
        break;
      }
#endif
      default:
        break;
    }
  }
}

void init_mode()
{
  uint8_t i;

  robot_init();            // robot.c -> initialisation des segments

#ifdef HAS_LOG
  log_pos = 0;
#endif

#ifdef HARDWARE_V3
  // Sets current accumulators to zero
  for (i = 0; i < MOD_COUNT; i++)
  {
    set_reg_value_w(mod_addr[i], MREG16_BATT_ACC, 0);       // utiliter ???
  }
  // Initzializations
  //set_color_i(7, 6);
  set_rgb(0, 0xFF, 0);//0x80
  for(i=0;i<MOD_COUNT;i++)
  {
    start_pid(mod_addr[i]);
    set_reg_value_dw(mod_addr[i], MREG32_LED, 0x00ff00);//0x00ffffff
    set_reg_value_b(mod_addr[i], 3, 64);    // allume LED de tracking
  }
/*
#ifdef THREE_LEDS
  set_color_el(1,0,0);
  set_color_el(2,0,0);
  set_color_el(3,0,0);
  set_color_el(4,0,0);
  set_color_el(5,0,0);
#endif
*/
#endif
  // Return to idle mode
  reg8_table[REG8_MODE] = IMODE_IDLE;
}

void stop_mode()
{
  uint8_t i;
  for (i=0; i<MOD_COUNT; i++)
  {
    bus_set(mod_addr[i], MREG_SETPOINT, 0);
  }
  pause(HUNDRED_MS * 3);
  for (i=0; i<MOD_COUNT; i++)
  {
    bus_set(mod_addr[i], MREG_MODE, MODE_IDLE);
    set_reg_value_dw(mod_addr[i], MREG32_LED, 0);
    set_reg_value_b(mod_addr[i], 3, 0);    // eteint LED de tracking
  }
  reg8_table[REG8_MODE] = IMODE_IDLE;
}

void bus_test_mode()
{
  uint8_t i;

  while (reg8_table[REG8_MODE] == IMODE_BUS_TEST)
  {
    for (i = 0; i<MOD_COUNT; i++) {
#ifdef HARDWARE_V3
      set_reg_value_dw(mod_addr[i], REG32_LED, 0x002A0A);
#else
      bus_set(mod_addr[i], MREG_HW_OPTIONS, HWO_LED);
#endif
    }
#ifdef HARDWARE_V3
    set_color_i(11, 4);
#else
    set_led(1);
#endif
    pause(HUNDRED_MS);
    for (i = 0; i<MOD_COUNT; i++) {
#ifdef HARDWARE_V3
      set_reg_value_dw(mod_addr[i], REG32_LED, 0);
#else
      bus_set(mod_addr[i], MREG_HW_OPTIONS, 0);
#endif
    }
#ifdef HARDWARE_V3
    set_color(0);
#else
    set_led(0);
#endif
    pause(HUNDRED_MS);

  }

  for (i = 0; i<MOD_COUNT; i++) {
#ifdef HARDWARE_V3
      set_reg_value_dw(mod_addr[i], REG32_LED, 0);
#else
      bus_set(mod_addr[i], MREG_HW_OPTIONS, 0);
#endif
  }
}

void pre_bootloader_mode()
{
#ifdef HARDWARE_V3
  set_rgb(15, 3, 0);
  i2c_set(RGB_ADDR, 8, 0x3F);   // ensures group blinking/PWM is enabled on pins
  i2c_set(RGB_ADDR, 1, 0x25);   // select blinking function instead of group PWM
  i2c_set(RGB_ADDR, 6, 127);    // blinking duty cycle 127/255 (symmetric blink)
  i2c_set(RGB_ADDR, 7, 10);     // set blinking period to 0.45 s  (10+1)/24 s
#else
  set_led(1);
#endif
  while (reg8_table[REG8_MODE] == IMODE_WILL_BOOTLOAD);
#ifdef HARDWARE_V3
  set_rgb(0, 0, 0);
  init_rgb_led();               // removes the blinking...
#else
  set_led(0);
#endif
}

#ifdef HARDWARE_V3
void charger_mode()
{
#ifdef EXT_ALIM
#else
  uint8_t c;
  int16_t cu;
  int32_t sm;

  if (get_ext_voltage()<=225) return;
  set_color_i(0, 4);       // set medium intensity
  while (get_ext_voltage() > 225 && reg8_table[REG8_MODE] != IMODE_WILL_BOOTLOAD) {   // abt. 8 V
    sm = 0;
    for (c=0; c<16; c++) {
      cu = get_batt_current();
      sm += cu;
    }
    sm = sm / 16;
    cu = sm;
    if (cu>20) set_color(4);         // red:    > 60 mA (C/10)
    if (cu>6 && cu<18) set_color(8); // orange: 10...60 mA (+ hysteresis)
    if (cu<4) set_color(2);          // green:  < 10 mA
  }
  set_color_i(11, 0);
#endif
}

uint8_t test_charger_mode()
{
  return (get_ext_voltage()>225);
}

#endif

#ifdef HARDWARE_V2

void charger_mode()
{
  uint8_t i;

  reg32_table[REG32_LED] = LED_MANUAL;
  for (i = 0; i < MOD_COUNT; i++) {
    bus_set(mod_addr[i], MREG_EXT_DEVICE, 1);
  }
  while (reg8_table[REG8_MODE] == IMODE_BATTERY_CHG) {
    set_led(1);
    pause(HUNDRED_MS * 8);
    set_led(0);
    pause(HUNDRED_MS * 8);
  }
  for (i = 0; i < MOD_COUNT; i++) {
    bus_set(mod_addr[i], MREG_EXT_DEVICE, 0);
  }
}

uint8_t test_charger_mode()
{
  return 0;
}

#endif

void cpg_mode(void)
{
  // Basic variables
  uint8_t i,j;
#ifdef HAS_SD
  uint16_t nbre_pos = 0;
  float time_read = 0;
#endif
  uint32_t dt,cycletimer;
  float a_r;
  float time = 0, deltat, coupling;
  
  //Simplifiers
  float coupling_strength;

  // General Amplitude, Phase and Offset of each fin
  float ampl = 0.0;
  float freq = 0.0;
  float nwave = 0.0;
  float nwave_current = 0.0;
  float dphi = 0.0;
  float	off_turn = 0.0;
  float	ampl_r = 0.0;
  float amplh = 0.0;
  float amplc = 0.0;
  
#ifdef HAS_LOG  
  int log_started = 0;
#endif

#ifdef EIGHT_ELEM
  // Output setpoint for body elements
  static int8_t setpoint[SEG_COUNT] = {0};

  // Oscillator phase variables
  static float osc_phi[OSC_COUNT] = {0};//{-EPS,EPS,EPS,-EPS,EPS,EPS,EPS,-EPS,EPS,EPS,EPS,EPS,EPS,EPS};
  osc_phi[0] = -EPS;
  osc_phi[SEG_COUNT] = -EPS;     // was EPS unlike in original table above
  for (i = 1; i<SEG_COUNT; i++){
    osc_phi[i] = EPS;
    osc_phi[i + SEG_COUNT] = EPS;
  }
  static float osc_dphi[OSC_COUNT] = {0};

  // Oscillator amplitude variables
  static float osc_r[OSC_COUNT] = {0};
  static float osc_dr[OSC_COUNT] = {0};
  static float osc_ddr[OSC_COUNT] = {0};

  // Connection table: senders and weights
  //static uint8_t osc_sender[OSC_COUNT][MAX_CONN_COUNT];
  static float osc_w[OSC_COUNT][OSC_COUNT] = {{0}};
  static float osc_wphi[OSC_COUNT][OSC_COUNT] = {{0}};
#endif

#ifdef THREE_ELEM
  // Output setpoint for body elements
  static int8_t setpoint[SEG_COUNT]={0};

  // Oscillator phase variables
  static float osc_phi[OSC_COUNT]= {-EPS,EPS,-EPS,EPS};
  static float osc_dphi[OSC_COUNT]={0};

  // Oscillator amplitude variables
  static float osc_r[OSC_COUNT]=   {0};
  static float osc_dr[OSC_COUNT]=  {0};
  static float osc_ddr[OSC_COUNT]=  {0};

  // Connection table: senders and weights
  //static uint8_t osc_sender[OSC_COUNT][MAX_CONN_COUNT];
  static float osc_w[OSC_COUNT][OSC_COUNT]={0};

  static float osc_wphi[OSC_COUNT][OSC_COUNT]={0};
#endif

#ifdef HAS_SD
  char				file_position[50];
  char				file_setpoint[50];
  uint8_t				id_log, id_file;
  int8_t				ret;
  uint32_t			n;
#endif
#ifdef HAS_SD
  id_log=reg8_table[REG8_ID_LOG];
  id_file=reg8_table[REG8_ID_FILE];
  // Init SD Card:
  if (efs_init(&sd_efs, 0) != 0) set_color(4);
  // file pos name:
  sprintf(file_position, "amph/P%d%d.txt",id_log,id_file);
  // file setpoint name:
  // sprintf(file_setpoint, "%s/S%d%d.txt","amph",id_log,id_file);
  // create a new directory:
  mkdir(&sd_efs.myFs, "amph");
  // open the file position:
  ret=file_fopen(&sd_file_position, &sd_efs.myFs, file_position, 'w');
  if (ret==-2)
  {
    rmfile(&sd_efs.myFs, file_position);
    file_fopen(&sd_file_position, &sd_efs.myFs, file_position, 'w');
  }
  // open the file setpoint:
  // ret=file_fopen(&sd_file_setpoint, &sd_efs.myFs, file_setpoint, 'w');
  // if (ret==-2)
  // {
  //	rmfile(&sd_efs.myFs, file_setpoint);
  //	file_fopen(&sd_file_setpoint, &sd_efs.myFs, file_setpoint, 'w');
  // }
#endif

  coupling_strength=DECODE_PARAM_8(reg8_table[REG8_COUPLING],MIN_COUPLING,MAX_COUPLING);
  a_r=DECODE_PARAM_8(reg8_table[REG8_TRANS_SPEED],MIN_TRANS_SPEED,MAX_TRANS_SPEED);

  // Coupling matrix init

  for (i = 0; i < OSC_COUNT; i++) {
    for (j = 0; j < OSC_COUNT; j++) {
      if ((j == i+1) && (j != SEG_COUNT)) {
        osc_w[i][j] = coupling_strength;                                                                   //
      } else if ((j == i-1) && (j != SEG_COUNT)) {
        osc_w[i][j] = coupling_strength;
      } else if ((j == SEG_COUNT+i)) {
        osc_w[i][j] = coupling_strength;
      } else if ((j == i-SEG_COUNT)) {
        osc_w[i][j] = coupling_strength;
      }
    }
  }

  // Timer init:
  initSysTime();
  cycletimer = getSysTICs() - TEN_MS;

#ifdef HAS_LOG  
  set_rgb(0, 0, 0);
#endif

  while (reg8_table[REG8_MODE] == IMODE_CPG_MODE)
  {
    nwave = DECODE_PARAM_8(reg8_table[REG8_NWAVE],MIN_NWAVE,MAX_NWAVE);
    dphi = nwave * M_TWOPI/(SEG_COUNT);
    if (nwave != nwave_current) {
      for (i = 0; i < OSC_COUNT; i++) {
        for (j = 0; j < OSC_COUNT; j++) {
          if ((j == i+1) && (j != SEG_COUNT)) {
            osc_wphi[i][j] = -dphi;
          } else if ((j == i-1) && (j != SEG_COUNT)) {
            osc_wphi[i][j] = dphi;
          } else if ((j == SEG_COUNT+i)) {
            osc_wphi[i][j] = M_PI;
          } else if ((j == i-SEG_COUNT)) {
            osc_wphi[i][j] = M_PI;
          }
        }
      }
      nwave_current = nwave;
    }
    amplc = DECODE_PARAM_8(reg8_table[REG8_AMPLC],MIN_AMPLC,MAX_AMPLC)/360.0*M_TWOPI;
    amplh = DECODE_PARAM_8(reg8_table[REG8_AMPLH],MIN_AMPLH,MAX_AMPLH)/360.0*M_TWOPI;
    freq = 1.4*1.5*DECODE_PARAM_8(reg8_table[REG8_FREQ],MIN_FREQ,MAX_FREQ);
    off_turn = DECODE_PARAM_8(reg8_table[REG8_TURN],MIN_TURN,MAX_TURN);
    //Calculate the current time
    dt = getElapsedSysTICs(cycletimer);
    cycletimer = getSysTICs();
    deltat = (float) dt / sysTICSperSEC;
    time += deltat;

    //Compute the derivatives
    for (i = 0; i < OSC_COUNT; i++) {
      coupling = 0.0;
      for (j = 0; j < OSC_COUNT; j++) {
        coupling += osc_w[i][j] * (osc_r[j]*sinlut(osc_phi[j] - osc_phi[i] - osc_wphi[i][j]));
      }
      osc_dphi[i] = (M_TWOPI * freq + coupling);
      if (i < SEG_COUNT) {
        ampl = amplh + (amplc-amplh) / (SEG_COUNT) * (i+1);
      } else {
        ampl = amplh + (amplc-amplh) / (SEG_COUNT) * (i-SEG_COUNT+1);
      }

#ifdef STEERING_CONTROL
      if (i<SEG_COUNT) {
        ampl_r = (ampl + ampl * off_turn) / 2.0;
      } else {
        ampl_r = (ampl - ampl * off_turn) / 2.0;
      }
#else
      ampl_r = ampl / 2.0;
#endif
      osc_ddr[i] = a_r * (0.25 * a_r * (ampl_r-osc_r[i]) - osc_dr[i]);
    }

    // Euler integration
    for (i=0; i < OSC_COUNT; i++) {
      osc_phi[i] += osc_dphi[i] * deltat;
      osc_dr[i]  += osc_ddr[i]  * deltat;
      osc_r[i]   += osc_dr[i]   * deltat;
    }

    // Keeps the phase between -2Pi and +2Pi, to avoid loops in coslut
    for (i = 0; i < OSC_COUNT; i++) {
      if (osc_phi[i] > M_TWOPI) {
        osc_phi[i] -= M_TWOPI;
      } else if (osc_phi[i] < -M_TWOPI) {
        osc_phi[i] += M_TWOPI;
      }
    }

    //Calculate Output
    for (i = 0; i < SEG_COUNT; i++) {
#ifdef HAS_LOG
      if (log_pos < LOG_SIZE || (reg8_table[REG8_FLAGS] & FLAG_NO_LOGSTOP))
#endif
      setpoint[i] = RAD_TO_OUTPUT_BODY((osc_r[i+SEG_COUNT]*(1.0+coslut(osc_phi[i+SEG_COUNT])) - osc_r[i]*(1.0+coslut(osc_phi[i]))));
#ifdef HAS_LOG      
      if (!log_started && setpoint[i] != 0) {
        log_started = 1;
        set_rgb(0, 0xFF, 0);
      }
#endif      
      bus_set(mod_addr[i], MREG_SETPOINT, setpoint[i]);
    }

    for (i = 0; i < SEG_COUNT; i++) {
      positions[i] = bus_get(mod_addr[i], MREG_POSITION, &flag_can);
      if (flag_can) {
        positions[i] = 127;
      }
    }

#ifdef HAS_LOG
    if (log_started) {
      if (log_pos < LOG_SIZE) {
        for (i = 0; i < SEG_COUNT; i++) {
          uint16_t pos = log_pos * SEG_COUNT;
          pos_buff[pos + i] = positions[i];
          set_buff[pos + i] = setpoint[i];
        }
        delta_t_buff[log_pos] = (uint8_t) (1000 * deltat);
        log_pos++;
      } else {
        set_rgb(0, 0xFF, 0xFF);
      }
    }
#endif

#ifdef HAS_SD
    //Read positions
    if (time>=time_read)
    {
      for(i=0;i<SEG_COUNT;i++)
      {
        pos_buff[nbre_pos*7+i]=bus_get(mod_addr[i], MREG_POSITION, &flag_can);
      }
      time_buff[nbre_pos]= (uint32_t) (time*1000);
      nbre_pos++;
      time_read+=0.04;
    }
#endif
  }  // end CPG loop
  
#ifdef HAS_SD

  // setpoint:
  for (k=0; k<nbre_pos-1; k++)
  {
    set_color(4);
    n=sprintf (sd_buff, "%d %d %d %d %d %d %d %d\n",
        time_buff[k],
        pos_buff[k*7+0],
        pos_buff[k*7+1],
        pos_buff[k*7+2],
        pos_buff[k*7+3],
        pos_buff[k*7+4],
        pos_buff[k*7+5],
        pos_buff[k*7+6]);
    file_write(&sd_file_position, n, sd_buff);
    set_color(2);
  }

  // close file position:
  file_fclose(&sd_file_position);
  // close file setpoint:
  //file_fclose(&sd_file_setpoint);
  // correctly close the filesystem:
  fs_umount(&sd_efs.myFs);
  set_color_i(7, 6);
#endif
  // Straight configuration:
  for (i=0; i<SEG_COUNT; i++)
    {
    bus_set(mod_addr[i], MREG_SETPOINT, 0);
    }
  // Return to idle mode
  reg8_table[REG8_MODE] = IMODE_IDLE;
}

void main_mode_loop()
{
  reg8_table[REG8_MODE] = IMODE_IDLE;

  while (1)
  {
    // Verifies if the charging mode has to be enabled
    if (test_charger_mode()) {
      reg8_table[REG8_MODE] = IMODE_BATTERY_CHG;
    }

    switch(reg8_table[REG8_MODE])
    {
      case IMODE_IDLE:
        break;
      case IMODE_INIT:
        init_mode();
        break;
      case IMODE_BATTERY_CHG:
        charger_mode();
        if (reg8_table[REG8_MODE] == IMODE_BATTERY_CHG) {
          reg8_table[REG8_MODE] = IMODE_IDLE;
        }
        break;
      case IMODE_STOP:
        stop_mode();
        break;
      case IMODE_BUS_TEST:
        bus_test_mode();
        break;
      case IMODE_WILL_BOOTLOAD:
        pre_bootloader_mode();
        break;
      case IMODE_CPG_MODE:
        cpg_mode();
        break;
      default:
        reg8_table[REG8_MODE] = IMODE_IDLE;
    }
  }
}

#include <iostream>
#include <fstream>
#include <cstdio>
#include "../../armbl/remregs.h"

using namespace std;

const uint16_t REG_INTF_VER = 0x3C0;       ///< radio interface firmware version
const uint16_t REG_INTF_CH = 0x3C1;        ///< radio interface channel number
const uint16_t REG_RWL_VER = 0x3E0;        ///< remote radio firmware version
const uint16_t REG_BL_CTRL = 0x3E2;        ///< bootloader control register (for reboot)
const uint16_t REG8_MODE = 0;              ///< robot mode
const uint16_t REG16_LOG_POS = 2;          ///< log pointer register
const uint16_t REGMB_LOG_ENTRY = 4;        ///< log data

const uint8_t REQ_LOCAL_INTF_VERSION = 5;        ///< required firmware version for local radio interface
const uint8_t REQ_REMOTE_INTF_VERSION = 0x49;    ///< required firmware version for remote radio interface

const int BODY_GEARBOX_RATIO = 135;
const int BODY_POS_DIVIDER_VAL = 128;
const int ENCODER_RESOLUTION = 512;

const int UNITS_PER_TURN_BODY = ((ENCODER_RESOLUTION * BODY_GEARBOX_RATIO) / BODY_POS_DIVIDER_VAL);

float pos_to_deg(const int8_t pos)
{
  return (pos * 360L) / (float) UNITS_PER_TURN_BODY;
}

bool init_radio_interface(const char* port_name, const uint8_t channel, CRemoteRegs& regs)
{
  if (!regs.open(port_name, 57600)) {
    return false;
  }
  if (!regs.sync()) {
    cerr << "Interface synchronization failed!" << endl;
    return false;
  }
  // Tests if the radio interface is there and has the correct firmware
  if (regs.get_reg_b(REG_INTF_VER) != REQ_LOCAL_INTF_VERSION) {
    cerr << "Wrong wireless interface firmware detected." << endl;
    return false;
  }

  // Sets the channel number on the radio interface
  regs.set_reg_b(REG_INTF_CH, channel);

  // Verifies if the communication with the remote radio works, and checks
  // that it has the right firmware
  uint8_t ver;
  if (!regs.get_reg_b(REG_RWL_VER, ver)) {
    cerr << "Unable to communicate with the remote module." << endl;
    return false;
  } else if (ver != REQ_REMOTE_INTF_VERSION) {
    cerr << "Wrong remote wireless firmware detected (" << ver << ")." << endl;
    return false;
  }

  return true;
}

bool retry_write_w(CRemoteRegs& regs, const uint16_t addr, const uint16_t data)
{
  for (int i(0); i < 5; i++) {
    if (regs.set_reg_w(addr, data)) {
      return true;
    }
  }
  return false;
}

bool retry_get_mb_entry(CRemoteRegs& regs, const uint16_t pos, uint8_t* data, uint8_t& len)
{
  for (int i(0); i < 5; i++) {
    if (regs.get_reg_mb(REGMB_LOG_ENTRY, data, len)) {
      return true;
    }
    if (!retry_write_w(regs, REG16_LOG_POS, pos)) {
      return false;
    }
  }
  return false;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " filename.csv" << endl;
    return 1;
  }

  CRemoteRegs regs;

  if (!init_radio_interface("COM1", 126, regs)) {
    return 1;
  }

  uint16_t count;
  if (!regs.get_reg_w(2, count)) {
    cerr << "Could not retrieve log entry count." << endl;
    return 1;
  }

  if (count == 0xffff) {
    cerr << "Log functionality is not supported by the robot's firmware." << endl;
    return 1;
  } else if (count == 0) {
    cout << "The robot's log is empty." << endl;
    return 0;
  }

  ofstream f(argv[1]);
  if (!f.is_open()) {
    cerr << "Could not open " << argv[1] << " for writing." << endl;
    return 1;
  }

  regs.set_reg_b(REG8_MODE, 0);  // make sure we exit CPG mode so the log is not corrupted

  cout << "Retrieving " << count << " log entries... " << flush;

  // reset log pointer
  if (!retry_write_w(regs, REG16_LOG_POS, 0)) {
    cout << "error." << endl;
    cerr << "Could not reset log pointer." << endl;
    if (!retry_write_w(regs, REG16_LOG_POS, count)) {
      cerr << "Communication broken, unable to restore remote log pointer." << endl;
    }
    f.close();
    unlink(argv[1]);
    return 1;
  }

  uint32_t time(0);

  for (uint16_t i(0); i < count; i++) {
    printf("%3d%%\b\b\b\b", (100L * i / count));
    fflush(stdout);

    uint8_t data[29], len;
    if (!retry_get_mb_entry(regs, i, data, len)) {
      cout << "error." << endl;
      cerr << "Communication error while retrieving log data. Please retry." << endl;
      if (!retry_write_w(regs, REG16_LOG_POS, count)) {
        cerr << "Communication broken, unable to restore remote log pointer." << endl;
      }
    }
    time += data[0];
    f.precision(3);
    f << fixed << time / 1000.0;
    f.precision(1);
    for (int j(1); j < len; j++) {
      f << "," << pos_to_deg((int8_t) data[j]);
    }
    f << endl;
  }

  cout << "download complete." << endl;
  if (regs.get_reg_w(REG16_LOG_POS) != count) {
    cerr << "Consistency error: the remote log pointer doesn't match the entry count." << endl;
    cerr << "Retrieved pointer: " << regs.get_reg_w(REG16_LOG_POS) << endl;
    return 1;
  } else {
    return 0;
  }

}

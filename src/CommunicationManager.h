#ifndef COMMUNICATION_MANAGER
#define COMMUNICATION_MANAGER
#include <Arduino.h>

#define WRITE_SERVO_COMMAND 0
#define READ_SERVO_COMMAND 1
#define BUFF_SIZE 4000

class SerialServoController;
class CommunicationManager{
    private:
        bool processMoveCommand = false;
        bool processPrintCommand = false;
        bool readingMoveCommand = false;
        bool readingPrintCommand = false;
        char moveCommandBuffer[BUFF_SIZE];
        char printCommandBuffer[BUFF_SIZE];
        const char startMarker = '<';
        const char endMarker = '>';
        SerialServoController* servoController;
        unsigned int bytesRecvd = 0;
        boolean readInProgress = false;

char messageFromPC[BUFF_SIZE] = {0};

    public:
        boolean newDataFromPC = false;
        double lastCommand[6];
        CommunicationManager();
        void begin();
        void loop();
        void getDataFromPC();
        void parseData();
        void registerServoController(SerialServoController* servos);
};
#endif
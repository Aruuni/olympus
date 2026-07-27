#include "RLInterface.h"
#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "transportlayer/tcp/TcpPaced.h"
#include <algorithm>
#include <numeric>
#include <ostream>
#include "CleanSlate.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(CleanSlate); // Lets omnet see and use this class

CleanSlate::CleanSlate():
    TcpPacedNoCC(), RLInterface() {
    if (debug) cout << "\tCleanSlate: Constructor called!";
}

CleanSlate::~CleanSlate() {
    if (debug) cout << "\tCleanSlate: Destructor method called. Goodbye.";
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
}

// Called during sim initialization
void CleanSlate::initialize() {
    if (debug) cout << "\tCleanSlate initialize()" << endl;
    this->delayCoefficient = this->conn->getTcpMain()->par("delayCoefficient");
    this->lossCoefficient = this->conn->getTcpMain()->par("lossCoefficient");
    this->fixedIntervals = this->conn->getTcpMain()->par("fixedIntervals");
    this->fixedIntervalDuration = this->conn->getTcpMain()->par("fixedIntervalDuration");
    this->maxRLSteps = this->conn->getTcpMain()->par("maxRLSteps");
    this->takeActions = this->conn->getTcpMain()->par("takeActions");
    this->debug = this->conn->getTcpMain()->par("printDebugMessages");

    // provide the RLInterface with a cComponent API (for signals)
    RLInterface::setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    RLInterface::initialise();
    TcpPacedNoCC::initialize();

    // Signals
    throughputSignal = owner->registerSignal("throughput");
    actionSignal = owner->registerSignal("action");
    TcpPacedConnection* pacedConn = dynamic_cast<TcpPacedConnection *>(conn);
    pacedConn->subscribe(pacedConn->retransmissionRateSignal, (cListener*) this);
}

// INET method, called after connection is established.
void CleanSlate::established(bool active) {
    if (debug) cout << "\tCleanSlate: established()" << endl;
    TcpPacedNoCC::established(active);
    
    // Only register this as an RL agent if this is a client
    if (active) {
        this->isActive = active;

        // Set the RayNet ID of this agent and register with the Broker
        RLInterface::setStringId("CleanSlate");
        cObject* simtime = new cSimTime(0); // Used to contain initial step length, now deprecated.
        owner->emit(this->registerSig, stringId.c_str(), simtime); 

        // Finally, schedule this agent's first RL step (first step uses fixedIntervalDuration, even if fixedIntervals==false)
        RLInterface::scheduleNextStep(this->fixedIntervalDuration);
    }
}

void CleanSlate::connectionClosed() {
    if (debug) cout << "\tCleanSlate: connectionClosed()" << endl;
    TcpPacedNoCC::connectionClosed();

    // Temporary flow-stop guard: prevent closed flows from keeping a usable cwnd.
    if (state) {
        state->snd_cwnd = 0;
    }

    done = true;
    if (isActive) {
        RLInterface::terminate();
    }
}

// Return an observation to the broker, based on the current state
std::optional<ObsType> CleanSlate::computeObservation(){
    if (debug) cout << "\tCleanSlate: computeObservation()" << endl; 
    double lastIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    this->delta_snd_max = state->snd_max - this->last_snd_max;
    this->delta_snd_una = state->snd_una - this->last_snd_una;
    this->delta_ack_cnt = state->ack_cnt - this->last_ack_cnt;
    
    // Initialize empty obs to populate as we go
    double obs[7] = {0,0,0,0,0,0,};

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->cleanSlateThroughput = this->bytesDelivered / lastIntervalDuration;
    this->cleanSlateMaxThroughput = std::max(this->cleanSlateMaxThroughput, this->cleanSlateThroughput);
    if (this->cleanSlateMaxThroughput == 0.0) {
        obs[0] = obs[1] = obs[2] = 0.0;
    } else {
        // Throughput
        obs[0] = this->cleanSlateThroughput / this->cleanSlateMaxThroughput;

        // Pacerate
        double paceRate = (1.0/dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl()) * (double) state->snd_mss;
        obs[1] = std::min(10.0, paceRate / this->cleanSlateMaxThroughput);

        // Lossrate: What percentage of bytes sent this interval were retransmissions
        this->cleanSlateLossRate = 0.0;
        if (this->retransmissionRate > 0.0) {  // Avoid division by 0
            double transmissionRate = delta_snd_max/lastIntervalDuration; // How many non-retransmits occurred this interval in bytes/s
            this->cleanSlateLossRate = this->retransmissionRate / (this->retransmissionRate + transmissionRate); // What percentage of interval's sent data was retransmissions
        }
        obs[2] = this->retransmissionRate / this->cleanSlateMaxThroughput;
    }

    // ACKed: How many bytes were ACKed this interval
    this->cleanSlateACKTotal= this->bytesDelivered/(double)state->snd_mss;
    obs[3] = this->cleanSlateACKTotal /  state->snd_cwnd;
    
    // Interval Duration: How many seconds passed since last interval
    obs[4] = lastIntervalDuration;

    // Delay: Tracked in overridden method above. Only update the minimum if delay reports were received this interval.
    if (this->rttReportCount == 0 || state->srtt.dbl() == 0.0) {
        // No RTT reports were received this interval. Set all RTT related values to 0 to represent lack of data.
        this->cleanSlateDelayMetric = 0.0;    // Delay is unobserved, not 0. Do not report as optimal.
        obs[5] = 0.0;                   // SRTT
        obs[6] = 0.0;                   // Delay metric
    } else {
        // Observation is valid, compute the delay metric (0.0 is poor, 1.0 is optimal. Report as optimal if within the delay budget)
        if (state->srtt > this->cleanSlateMinDelay * this->delayCoefficient) {                                             
            this->cleanSlateDelayMetric = this->cleanSlateMinDelay * this->delayCoefficient / state->srtt;
        } else {
            this->cleanSlateDelayMetric = 1.0;
        }
        obs[5] = this->cleanSlateMinDelay / std::max(state->srtt.dbl(), this->cleanSlateMinDelay); // sRTT is lower than it should be at startup. The max prevents this obs from being > 1 when that happens.
        obs[6] = this->cleanSlateDelayMetric;
    }
    

    // Schedule next step if episode not done
    RLStepsTaken++;
    if (RLStepsTaken >= this->maxRLSteps && this->maxRLSteps > 0) {
        done = true;
    } else {
        scheduleNextStep(this->fixedIntervals ? this->fixedIntervalDuration : state->srtt.dbl());
    }
    
    // Debug prints
    if(debug) {
        cout << "-" << endl;
        cout << "! STEP: " << RLStepsTaken << " !" << endl;
        cout << "\tState:" << endl;
            cout << "\t\tsnd_una: " << state->snd_una << endl;
            cout << "\t\tdelta_snd_una: " << delta_snd_una << endl;
            cout << "\t\tcwnd: " << state->snd_cwnd << endl;
            cout << "\t\tsnd_max: " << state->snd_max << endl;
            cout << "\t\tSRTT: " << state->srtt << endl;
            cout << "\t\tack_cnt: " << state->ack_cnt << endl;
            cout << "\t\tdelta_ack_cnt: " << this->delta_ack_cnt << endl;
            cout << "\t\tACKTotal: " << this->cleanSlateACKTotal << endl;
            cout << "\t\tpacketsDelivered" << this->bytesDelivered/state->snd_mss << endl;
            cout << "\t\trttReportCount: " << this->rttReportCount << endl;
            cout << "\t\tDone: " << done << endl;
        cout << "\tObservations:" << endl;
            cout << "\t\tThroughput: " << obs[0] << endl;
            cout << "\t\tPacerate: " << obs[1] << endl;
            cout << "\t\tLossRate: " << obs[2] << endl;
            cout << "\t\tAcksCount: " << obs[3] << endl;
            cout << "\t\tIntervalDuration: " << obs[4] << endl;
            cout << "\t\tRaw SRTT: " << obs[5] << endl;
            cout << "\t\tSRTT Metric: " << obs[6] << endl;
    } 

    // Plotting metrics
    owner->emit(throughputSignal, this->cleanSlateThroughput);

    return ObsType({
            obs[0],                   // Normalized throughput
            obs[1],               // Normalized pacerate
            obs[2],               // Normalized lossrate
            obs[3],              // Normalized ACKs count (maybe use tcp_cwnd? ask aiden)     
            obs[4],              // Monitor interval duration
            obs[5],             // Normalized SRTT (delay)
            obs[6]              // Normalized SRTT (possibly forgiven, if within the forgiveness window)
    });
}

RewardType CleanSlate::computeReward(){
    if (debug) cout << "\tCleanSlate: computeReward()" << endl;
    double reward;
    if (this->cleanSlateMaxThroughput == 0.0) {
        reward = 0.0;
    } else {
        reward = (this->cleanSlateThroughput-(this->lossCoefficient*this->cleanSlateLossRate))/this->cleanSlateMaxThroughput*this->cleanSlateDelayMetric;
    }
    if (debug) cout << "\t\tReward: " << reward << endl;
    return(reward);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void CleanSlate::decisionMade(ActionType action) {
    if (debug) cout << "\tCleanSlate: decisionMade()" << endl;

    // Compute new cwnd from the given action (cwnd *= 2^action)
    double multiplier = std::pow(2.0, (double) action);
    uint32_t newCwnd = ceil(((double) state->snd_cwnd) * multiplier);
    newCwnd = max(newCwnd, state->snd_mss);
    
    // Attempt to change cwnd and pacing rate
    if (this->takeActions && this->rttReportCount > 0 && newCwnd <= 1000000) {
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;

        // Update pacing rate
        if(state->snd_cwnd > 0) {
            double paceFactor;
            if (state->snd_cwnd < state->ssthresh/2) {
                paceFactor = 2;
            }
            else{
                paceFactor = 1.2;
            }
        uint32_t maxWindow = std::max(state->snd_cwnd, dynamic_cast<TcpPacedConnection*>(conn)->getBytesInFlight());
        double pace = state->srtt.dbl()/(((double) (maxWindow) / (double)state->snd_mss) * paceFactor);
        dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(pace);
        }
        owner->emit(actionSignal, multiplier); // Emit action for plotting
    } else {
        // Invalid. Skip this action entirely.
        if (debug) {
            cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
        } 
    }

    if (debug) {
        cout << "\t\t" << (this->takeActions) << endl;
        cout << "\t\t" << (this->rttReportCount > 0) << endl;
        cout << "\t\t" << (newCwnd < 1000000) << endl;
        cout << "-" << endl;
    }
}


void CleanSlate::resetStepVariables()
{
    if (debug) cout << "\t\tCleanSlate: resetStepVariables()" << endl;
    this->cleanSlateThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    this->cleanSlateLossRate=0.0;      // The average loss rate of packets over the last interval
    this->cleanSlateDelaySum=0.0;      // Sum of all RTT reports received over an interval 
    this->cleanSlateACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    this->bytesDelivered=0.0;
    this->rttReportCount=0; // The number of RTT values we have measured over the last interval
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->last_ack_cnt = state->ack_cnt;
    this->lastIntervalTime = simTime();
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void CleanSlate::cleanup()
{
    if (debug) cout << "\tCleanSlate: cleanUp()" << endl;
}

ObsType CleanSlate::getRLState(){
    if (debug) cout << "\tCleanSlate: getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType CleanSlate::getReward(){
    if (debug) cout << "\tCleanSlate: getReward()" << endl;
    // Deprecated, remove this later
}

bool CleanSlate::getDone() {
    return done;
    // Deprecated, remove this later (done is just set and checked directly)
}

// MARK: Cubic Methods
// ============================================================================
// These are copied directly from cubic.
// and a couple lines were added for tracking some extra information for CleanSlate to use.
// ============================================================================

// Called upon an ACK. Store the RTT info for averaging at the end of the interval.
void CleanSlate::rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) {
    TcpPacedNoCC::rttMeasurementComplete(tSent, tAcked);
    double packetRTT = (tAcked-tSent).dbl();
    this->cleanSlateDelaySum += packetRTT;
    this->rttReportCount += 1;

    
    this->cleanSlateMinDelay = std::min(this->cleanSlateMinDelay, packetRTT);
}

// Override to track bytes delivered
void CleanSlate::receivedDataAck(uint32_t firstSeqAcked) {
    TcpPacedNoCC::receivedDataAck(firstSeqAcked);
    this->bytesDelivered += state->snd_mss; // Number of bytes sent so far this interval
}

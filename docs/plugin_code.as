BonkStateManager bs;

float prev_speed = 0;

float bonkTargetThresh = 0.f; //najwazniejsze

const float BONK_THRESHOLD = 16.0f;
const uint PIPE_DEBOUNCE = 5000;
const int WHEEL_COUNT_ARR_SIZE = 10;

//parameters from settings
[Setting category="Bonk!s" min=0 max=100 name="Bonk threshold" description="How sensitive the Bonk! detection is. If you get many false positives, increase this value."]
float bonkThresh = 16.f;

[Setting category="Pipes" min=0 max=60000 name="Pipe debounce" description="Length (in ms) to cool down before making additional pipe sounds."]
uint pipeDebounce = 5000;

CSmPlayer@ LocalPlayer;

int notContactCheck(CSceneVehicleVisState@ visState, EPlugSurfaceMaterialId surface) {
        return 
            (visState.FLGroundContactMaterial != surface ? 1 : 0) +
            (visState.FRGroundContactMaterial != surface ? 1 : 0) +
            (visState.RLGroundContactMaterial != surface ? 1 : 0) +
            (visState.RRGroundContactMaterial != surface ? 1 : 0);
    }

void init() {
	bs = BonkStateManager();
	print("Creating State Menager");
}

bool step(float jerk, bool isBraking) {
	CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
	if (vis is null) {
		return false;
	}
	bool isBonk = false;

  	if (vis.RaceStartTime == 0xFFFFFFFF || vis.FrontSpeed == 0) { // in pre-race mode
		prev_speed = 0;
		return false;
	}

	auto mlf = MLFeed::GetRaceData_V3();
	auto plf = mlf.GetPlayer_V3(MLFeed::LocalPlayersName);
	if (plf.spawnStatus != MLFeed::SpawnStatus::Spawned || (plf.LastRespawnRaceTime - plf.CurrentRaceTime) > 0) {
		prev_speed = 0;
		return false;
	}
	
	float speed = getSpeed(vis);
	float curr_acc;

	try {
		curr_acc = Math::Max(0, (prev_speed - speed) * 1000);
		//print("curr_acc = " + curr_acc);
	} 
	catch {
		curr_acc = 0;
	}
	prev_speed = speed;
	
	if (speed < 0) {
		speed *= -1.f;
		curr_acc *= -1.f;
	}
	bonkTargetThresh = (bonkThresh + prev_speed * 1.5f);

	isBonk = curr_acc > bonkTargetThresh;
	
	if (jerk > -1.45f || isBraking) {
		isBonk = false;
	}
	bs.handleBonkCall(vis, isBonk);
	return isBonk;
}


float getSpeed(CSceneVehicleVisState@ vis) {
	return vis.WorldVel.Length();
}


class BonkStateManager{
    int idx = 0;

    bool bonk;
    bool pipe;

    vec3 prevVel;
    vec3 prevVelNorm;
    float prevVelLength;
    vec3 prevVdt;

    uint64 lastPipeTime = 0;

    int prevWheelContactCount;

    array<int> wheelContactCountArr(10);

    int pipeCountDown = -1;

    /* To be called once per frame. */ 
    void handleBonkCall(CSceneVehicleVisState@ visState, bool mainBonkDetect) {
        vec3 v = visState.WorldVel;
        float vLen = v.Length();

        int wheelContactCount = notContactCheck(visState, EPlugSurfaceMaterialId::XXX_Null);
        wheelContactCountArr[idx] = wheelContactCount;

        if (pipeCountDown > 0) {
            if (wheelContactCount == 0) {
                pipeCountDown -= 1;
            } else {
                pipeCountDown = -1;
            } 
        } else if (pipeCountDown == 0) {
            pipeCountDown = -1;
            lastPipeTime = Time::Now;
            return;
        }
        
        vec3 vdt = v - prevVel; 
        float vdtUp = Math::Dot(vdt, visState.Up);
        vdt = vdt - visState.Up * vdtUp;
        if (Math::Dot(vdt, visState.Dir) > 0) {
            vdt = vdt - visState.Dir * (Math::Dot(vdt, visState.Dir));
        }

        vec3 vdtdt = vdt - prevVdt;

        // Case: roofhit
        // Is the force opposite in direction to the up vector? 
        // Also we only want to roofhit when we are pointing down - otherwise it will be overdone and not funny
        // We also check to make sure we were falling for at least 10 frames beforehand, plus we start this countdown
        // to ensure that we don't touch the ground with any wheel for 3 frames after. 
        if (
            	(lastPipeTime < Time::Now - pipeDebounce) && 
            	(prevVelLength > 10) &&
            	(vLen > 3) && 
            	Math::Abs(vdtUp) > (vLen * 0.1) && 
            	pipeCountDown == -1 && 
            	(Math::Dot(visState.Up, vec3(0, -1, 0)) > 0.9) &&
            	sumWheelContactCountArr() == 0 && 
            	mainBonkDetect
           ) {
                pipeCountDown = 3;
            }
        
        prevVelLength = vLen;
        prevVel = v;
        prevWheelContactCount = wheelContactCount;
        prevVdt = vdt;
        idx = (idx + 1) % 10;
        return;
    }

    int sumWheelContactCountArr() {
        int r = 0;
        for (int i = 0; i < 10; i++) {
            r += wheelContactCountArr[i];
        }
        return r;
    }
}

bool FindLocalPlayer() {
    CGamePlayground@ Playground = GetApp().CurrentPlayground;
    if (Playground is null) return false;

    MwFastBuffer<CGameTerminal@> Terminals = Playground.GameTerminals;
    if (Terminals.Length == 0) return false;

    auto Player = cast<CSmPlayer@>(Terminals[0].ControlledPlayer);
    if (Player is null) {
        // Not yet loaded
        return false;
    }

    @LocalPlayer = Player;
    return true;
}

bool send_memory_buffer(Net::Socket@ sock, MemoryBuffer@ buf)
{
	if (!sock.Write(buf))
	{
		// If this fails, the socket might not be open. Something is wrong!
		print("INFO: Disconnected, could not send data.");
		return false;
	}
	return true;
}

// cast val to a float when necessary and append it to buf:
// "As the scripting engine has been optimized for 32 bit datatypes, using the smaller variants is only recommended for accessing application specified variables. For local variables it is better to use the 32 bit variant."
void append_float(MemoryBuffer@ buf, float val)
{
	buf.Write(val);
}

void append_bool(MemoryBuffer@ buf, bool val)
{
	if (val)
	{
		buf.Write(1.0f);
	}
	else
	{
		buf.Write(0.0f);

	}
}

void append_int(MemoryBuffer@ buf, int32 val)
{
	buf.Write(float(val));
}

void Main()
{
	print("TQC plugin: Main() started.");
	if (!FindLocalPlayer()) {
		print("TQC plugin: FindLocalPlayer() failed - not in a race/playground? Retrying on next load.");
		return;
	}
	auto ScriptAPI = cast<CSmScriptPlayer@>(LocalPlayer.ScriptAPI);

	init();
	print("TQC plugin: init done, starting server loop.");

	float prev_speed = 0;
    float speed = 0;
	float prev_acceleration = 0;
    float acceleration = 0;
	float jerk = 0;
	bool isBraking = false;
	bool isFinished = false;
    int _curCP = 0;
    int _curLap = 0;
	while(true)
	{
		CSceneVehicleVisState@ vehicle = VehicleState::ViewingPlayerState();

		CHmsCamera@ cam = Camera::GetCurrent();

		auto sock_serv = Net::Socket();
		if (!sock_serv.Listen("127.0.0.1", 9000)) {
			print("Could not initiate server socket.");
			return;
		}
		print(Time::Now + ": Waiting for incoming connection...");

		while(!sock_serv.IsReady()){
			yield();
		}
		print(Time::Now + ": Server socket ready");

		// Same pattern as working TMRL plugin: yield before Accept(), retry if null
		while(true)
		{
			yield();
			auto sock = sock_serv.Accept();
			if(sock is null) continue;
			print(Time::Now + ": Connected!");

		// OpenPlanet can store bytes in a MemoryBuffer:
		MemoryBuffer@ buf = MemoryBuffer(0);

		bool cc = true;
		uint lastSkipLog = 0;
		uint lastSendLog = 0;
		uint frameCount = 0;
		bool didLogLoopStart = false;
		while(cc)
		{
			CTrackMania@ app = cast<CTrackMania>(GetApp());
			if(app is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - app is null (not in TrackMania?)"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}
			CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
			if(playground is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - CurrentPlayground is null (no arena loaded?)"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}
			CSmArena@ arena = cast<CSmArena>(playground.Arena);
			if(arena is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - Arena is null"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}
			if(arena.Players.Length <= 0)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - arena.Players.Length <= 0"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}

			auto player = arena.Players[0];
			if(player is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - player is null"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}

			CSmScriptPlayer@ api = cast<CSmScriptPlayer>(player.ScriptAPI);
			if(api is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - ScriptAPI is null"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}
			if(vehicle is null)
			{
				if (Time::Now - lastSkipLog > 3000) { print("TQC plugin: skipping frame - vehicle (ViewingPlayerState) is null"); lastSkipLog = Time::Now; }
				yield();
				continue;
			}
			if (!didLogLoopStart) {
				print("TQC plugin: data loop started, sending frames.");
				didLogLoopStart = true;
			}
			auto race_state = playground.GameTerminals[0].UISequence_Current;

			speed = api.Speed;
      		acceleration = speed - prev_speed;
			jerk = acceleration - prev_acceleration;
      		prev_speed = speed;
			prev_acceleration = acceleration;
			isBraking = api.InputIsBraking;

            if(race_state == SGamePlaygroundUIConfig::EUISequence::Finish || race_state == SGamePlaygroundUIConfig::EUISequence::EndRound)
			{
				isFinished = true;
			}
			else
			{
				isFinished = false;
			}

			if(PlayerState::GetRaceData().PlayerState == PlayerState::EPlayerState::EPlayerState_Driving) 
			{
				auto info = PlayerState::GetRaceData().dPlayerInfo;
				_curCP = info.NumberOfCheckpointsPassed;
				_curLap = info.CurrentLapNumber;
			} 
			else 
			{
				_curCP = 0;
				_curLap = 0;
			}

			buf.Seek(0, 0);
			// Sending data
            append_float(buf, _curCP); // 0
            append_float(buf, _curLap);	

			append_float(buf, speed); // 2

			append_float(buf, api.Position.x); 
			append_float(buf, api.Position.y); // 4
			append_float(buf, api.Position.z);
			append_float(buf, api.InputSteer); // 6
			append_float(buf, api.InputGasPedal);
			append_bool(buf, isBraking); // 8

			append_bool(buf, isFinished);

			append_float(buf, acceleration); // 10
			append_float(buf, jerk);
			
      		append_float(buf, api.AimYaw); // 12
			append_float(buf, api.AimPitch);

			append_float(buf, vehicle.FLSteerAngle);
			append_float(buf, vehicle.FRSteerAngle); // 15

            append_float(buf, vehicle.FLSlipCoef); 
			append_float(buf, vehicle.FRSlipCoef); // 17	
			
			append_bool(buf, step(jerk, isBraking)); // isCrashed
			
			append_float(buf, api.EngineCurGear); // 19 int


			buf.Seek(0, 0);
			// 20 danych
	    	cc = send_memory_buffer(sock, buf);
			frameCount += 1;
			if (Time::Now - lastSendLog > 5000) {
				print("TQC plugin: sent " + frameCount + " frames so far.");
				lastSendLog = Time::Now;
			}

			yield();  // this statement stops the script until the next frame
		}
		print("TQC plugin: data loop ended (send failed or client disconnected).");
		sock.Close();
		// Keep sock_serv open; inner loop will yield() and Accept() next client
		}
	}
}
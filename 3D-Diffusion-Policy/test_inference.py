import zarr
import asyncio
import websockets
import numpy as np
import json
import argparse
import os

async def test_inference(zarr_path, url="ws://localhost:8000/ws", obs_steps=2):
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Check available keys in data
    print(f"Available keys in data: {list(root['data'].keys())}")
    
    payload_obs = {}
    
    # Define the keys we need based on the config/task
    keys_to_fetch = ['point_cloud', 'state']
    
    start_idx = 0
    end_idx = start_idx + obs_steps
    
    for key in keys_to_fetch:
        if key in root['data']:
            # Get data and convert to list
            data_slice = root['data'][key][start_idx:end_idx]
            
            # Key mapping for server: 'state' -> 'agent_pos'
            server_key = key
            if key == 'state':
                server_key = 'agent_pos'
                print(f"Mapping 'state' -> 'agent_pos' for server request")

            # Simple cast to list for JSON serialization
            payload_obs[server_key] = data_slice.tolist()
            print(f"Loaded {key}: shape {data_slice.shape}")
        else:
            print(f"Warning: Key {key} not found in zarr dataset, skipping.")

    request_data = {
        "observation": payload_obs
    }
    
    print(f"Connecting to {url}...")
    try:
        async with websockets.connect(url) as websocket:
            print("Sending request...")
            await websocket.send(json.dumps(request_data))
            
            print("Waiting for response...")
            response_text = await websocket.recv()
            result = json.loads(response_text)
            
            if 'action' in result:
                action = np.array(result['action'])
                print("\nSUCCESS!")
                print(f"Received Prediction Shape: {action.shape}")
                
                # Fetch Ground Truth Actions
                if 'action' in root['data']:
                    pred_len = action.shape[0]
                    # Align GT: start_idx + (obs_steps - 1)
                    gt_start = start_idx + (obs_steps - 1)
                    gt_end = gt_start + pred_len
                    
                    gt_action = root['data']['action'][gt_start:gt_end]
                    
                    print(f"GT Action Shape aligned: {gt_action.shape}")
                    
                    if gt_action.shape[0] < pred_len:
                        print(f"Warning: GT ends early! (GT len {gt_action.shape[0]} < Pred len {pred_len})")
                        action = action[:gt_action.shape[0]]

                    print("-" * 60)
                    print(f"Step 0 (Start) Comparison:")
                    print(f"Pred: {action[0]}")
                    print(f"GT  : {gt_action[0]}")

                    print(f"\nStep {len(action)-1} (End) Comparison:")
                    print(f"Pred: {action[-1]}")
                    print(f"GT  : {gt_action[-1]}")
                    
                    # RMSE
                    mse = np.mean((action - gt_action) ** 2)
                    rmse = np.sqrt(mse)
                    print(f"\nRMSE (over {len(action)} steps): {rmse:.6f}")
                    print("-" * 60)
                else:
                     print("Key 'action' not found in Zarr, cannot compare.")
            else:
                print(f"\nError in response: {result}")
            
    except Exception as e:
        print(f"\nConnection Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, 
                        default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
                        help="Path to the Zarr dataset file")
    parser.add_argument('--url', type=str, default="ws://localhost:8000/ws",
                        help="Inference server WebSocket URL")
    parser.add_argument('--steps', type=int, default=1, help="Number of observation steps to send")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
    else:
        # Run the async test
        asyncio.run(test_inference(args.zarr_path, args.url, args.steps))

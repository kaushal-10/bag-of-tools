"""
Optimized Zero-Copy mmap-based shared memory streamer.
Supports dynamic frame shape changes, JSON metadata, and sub-10ms latency.
"""
import numpy as np
import time
import json
import mmap
import os
import struct
import atexit

import multiprocessing as mp
import time
import cv2


class SharedMemoryStreamer:
    # Header: timestamp(double), frame_id(uint64), height(uint32), width(uint32), meta_size(uint32), write_lock(uint8)
    # The write_lock (B) is used as a fast cross-process spinlock to prevent tearing.
    HEADER_FORMAT = "<dQIIIB"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 29 bytes

    META_LEN_PREFIX_FORMAT = "<I"
    META_LEN_PREFIX_SIZE = struct.calcsize(META_LEN_PREFIX_FORMAT)  # 4 bytes

    def __init__(self, file_path: str, shape=(480, 640, 3), dtype=np.uint8, create_if_missing=False, meta_size: int = 4096):
        # 🚨 PERFORMANCE ENFORCEMENT: Force use of RAM disk on Linux
        if os.name == 'posix' and not file_path.startswith('/dev/shm/'):
            print(f"⚠️ WARNING: {file_path} is not in /dev/shm/. You are likely writing to a physical disk! This will cause massive latency.")

        self.file_path = file_path
        self.dtype = np.dtype(dtype)
        self.channels = int(shape[2]) if len(shape) == 3 else 1
        self.create_if_missing = create_if_missing
        self.meta_size = int(meta_size)

        self.shape = tuple(shape)
        self.header_size = self.HEADER_SIZE
        self.frame_bytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.total_size = self.header_size + self.frame_bytes + self.meta_size

        self.mmap_obj = None
        self.file_obj = None
        self._closed = False
        self._shared_array_view = None # Cache the NumPy zero-copy view

        self._connect()

        if create_if_missing:
            atexit.register(self._cleanup)

    def _update_zero_copy_view(self):
        """Creates a NumPy array that points DIRECTLY to the mmap memory. No copying!"""
        if self.mmap_obj:
            self._shared_array_view = np.ndarray(
                self.shape, 
                dtype=self.dtype, 
                buffer=self.mmap_obj, 
                offset=self.header_size
            )

    def _connect(self):
        try:
            if os.path.exists(self.file_path):
                self.file_obj = open(self.file_path, "r+b")
                actual_size = os.path.getsize(self.file_path)

                header_map = mmap.mmap(self.file_obj.fileno(), self.header_size, access=mmap.ACCESS_READ)
                try:
                    header_bytes = header_map.read(self.header_size)
                    if len(header_bytes) == self.header_size:
                        timestamp, frame_id, height, width, meta_size, _ = struct.unpack(self.HEADER_FORMAT, header_bytes)
                        if height > 0 and width > 0:
                            self.shape = (int(height), int(width), self.channels)
                            self.meta_size = int(meta_size)
                            self.frame_bytes = int(height) * int(width) * self.channels * self.dtype.itemsize
                            self.total_size = self.header_size + self.frame_bytes + self.meta_size
                finally:
                    header_map.close()

                self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.total_size)
                self._update_zero_copy_view()
                
            elif self.create_if_missing:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                self.file_obj = open(self.file_path, "w+b")
                self.file_obj.truncate(self.total_size)
                self.file_obj.flush()
                self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.total_size)
                self._reset_memory()
                self._update_zero_copy_view()
                
        except Exception as e:
            print(f"Error connecting: {e}")
            self.mmap_obj = None

    def _reset_memory(self):
        if self.mmap_obj is None: return
        self.mmap_obj.seek(0)
        self.mmap_obj.write(b'\x00' * self.total_size)
        header = struct.pack(self.HEADER_FORMAT, 0.0, 0, self.shape[0], self.shape[1], self.meta_size, 0)
        self.mmap_obj.seek(0)
        self.mmap_obj.write(header)
        self.mmap_obj.flush()

    def _set_write_lock(self, is_writing: bool):
        """Sets the last byte of the header to 1 (writing) or 0 (free)."""
        # The lock byte is exactly at index 28 (HEADER_SIZE - 1)
        self.mmap_obj[self.HEADER_SIZE - 1] = 1 if is_writing else 0

    def write(self, frame: np.ndarray, metadata: dict = None):
        if self._closed or self.mmap_obj is None: return

        # Handle shape changes dynamically
        if frame.shape != self.shape:
            if self.create_if_missing:
                self.close()
                self.shape = frame.shape
                self.frame_bytes = int(np.prod(self.shape) * self.dtype.itemsize)
                self.total_size = self.header_size + self.frame_bytes + self.meta_size
                self._connect()
            else:
                raise ValueError("Frame shape mismatch.")

        try:
            # 1. LOCK: Tell consumers we are writing
            self._set_write_lock(True)

            timestamp = time.time()
            frame_id = metadata.get('frame_id', 0) if metadata else 0

            # 2. Write Header (Keep lock as 1)
            header = struct.pack(self.HEADER_FORMAT, timestamp, int(frame_id), self.shape[0], self.shape[1], self.meta_size, 1)
            self.mmap_obj.seek(0)
            self.mmap_obj.write(header)

            # 3. FAST ZERO-COPY WRITE: Use numpy to copy memory block directly
            np.copyto(self._shared_array_view, frame)

            # 4. Write Metadata
            meta_json = json.dumps(metadata or {}).encode("utf-8")
            max_payload = max(0, self.meta_size - self.META_LEN_PREFIX_SIZE)
            meta_json = meta_json[:max_payload]

            meta_offset = self.header_size + self.frame_bytes
            self.mmap_obj.seek(meta_offset)
            self.mmap_obj.write(struct.pack(self.META_LEN_PREFIX_FORMAT, len(meta_json)))
            if meta_json:
                self.mmap_obj.write(meta_json)

            # 5. UNLOCK: Done writing
            self._set_write_lock(False)

        except Exception as e:
            self._set_write_lock(False)
            print(f"Error writing: {e}")

    def read(self, timeout=0.1):
        if self._closed or self.mmap_obj is None: return None, None

        start_time = time.time()
        
        # 1. SPINLOCK: Wait if producer is currently writing
        while self.mmap_obj[self.HEADER_SIZE - 1] == 1:
            if time.time() - start_time > timeout:
                return None, None # Timeout waiting for lock
            time.sleep(0.001) # Sleep 1ms to prevent CPU burn

        try:
            # 2. Read Header
            self.mmap_obj.seek(0)
            header_bytes = self.mmap_obj.read(self.header_size)
            timestamp, frame_id, height, width, meta_size, _ = struct.unpack(self.HEADER_FORMAT, header_bytes)

            if timestamp == 0: return None, None

            # Detect shape change
            if (int(height), int(width), self.channels) != self.shape:
                self.shape = (int(height), int(width), self.channels)
                self.meta_size = int(meta_size)
                self.frame_bytes = int(height) * int(width) * self.channels * self.dtype.itemsize
                self.total_size = self.header_size + self.frame_bytes + self.meta_size
                
                # Remap mapping object to new size
                self.mmap_obj.close()
                self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.total_size)
                self._update_zero_copy_view() # Update view for new size

            # 3. FAST ZERO-COPY READ: Return the direct view
            # Note: We return a reference to the array. If the user wants to keep 
            # a history of frames, they must explicitly .copy() it in their code.
            frame_view = self._shared_array_view

            # 4. Read Metadata
            meta_offset = self.header_size + self.frame_bytes
            self.mmap_obj.seek(meta_offset)
            raw_len_bytes = self.mmap_obj.read(self.META_LEN_PREFIX_SIZE)
            
            meta_len = struct.unpack(self.META_LEN_PREFIX_FORMAT, raw_len_bytes)[0]
            if 0 < meta_len <= (self.meta_size - self.META_LEN_PREFIX_SIZE):
                meta_json_bytes = self.mmap_obj.read(meta_len)
                meta = json.loads(meta_json_bytes.decode('utf-8', errors='replace'))
            else:
                meta = {}

            meta['shm_timestamp'] = timestamp
            meta['shm_frame_id'] = frame_id

            return frame_view, meta

        except Exception as e:
            print(f"Error reading: {e}")
            return None, None

    def close(self):
        self._closed = True
        try:
            if self.mmap_obj: self.mmap_obj.close()
            if self.file_obj: self.file_obj.close()
            if self.create_if_missing and os.path.exists(self.file_path):
                os.unlink(self.file_path)
        except Exception:
            pass

    def _cleanup(self):
        self.close()


def single_stream():
    import multiprocessing as mp
    import cv2
    import os
    import time
    import numpy as np

    def producer(shm_path, video_source, ready_event):
        print("[Producer] Starting up...")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"[Producer] Error: Could not open video source {video_source}")
            ready_event.set() # Unblock consumer so it can exit safely
            return

        # 1. Read the first frame to get the exact video dimensions
        ret, frame = cap.read()
        if not ret:
            print("[Producer] Error: Could not read first frame.")
            ready_event.set()
            return
            
        actual_shape = frame.shape
        print(f"[Producer] Video shape detected as: {actual_shape}")

        # 2. Create the Shared Memory block
        streamer = SharedMemoryStreamer(
            file_path=shm_path, 
            shape=actual_shape, 
            dtype=np.uint8, 
            create_if_missing=True
        )
        
        # Signal the consumer that the file is ready to be read
        ready_event.set()

        frame_id = 0
        try:
            while True:
                # Write current frame to memory
                streamer.write(frame, metadata={"frame_id": frame_id})
                frame_id += 1
                
                # Pace to ~30 FPS (remove this if using a live webcam)
                time.sleep(0.033)
                
                # Fetch the next frame
                ret, frame = cap.read()
                if not ret:
                    print("[Producer] Video finished.")
                    break
                    
        except KeyboardInterrupt:
            print("\n[Producer] Caught Ctrl+C, stopping gracefully.")
        finally:
            cap.release()
            streamer.close()
            print("[Producer] Exiting.")

    def consumer(shm_path, ready_event):
        print("[Consumer] Waiting for producer to initialize memory...")
        ready_event.wait()
        
        # Tiny delay to ensure OS file metadata is fully flushed
        time.sleep(0.1) 
        
        print("[Consumer] Memory ready, connecting...")

        streamer = SharedMemoryStreamer(
            file_path=shm_path, 
            create_if_missing=False
        )

        if streamer.mmap_obj is None:
            print("[Consumer] Could not connect to shared memory. Exiting.")
            return

        last_frame_id = -1

        try:
            while True:
                # The read() function handles the sub-millisecond spin-lock automatically
                frame_view, meta = streamer.read(timeout=1.0)
                
                if frame_view is not None:
                    current_frame_id = meta.get('shm_frame_id', -1)
                    
                    # Only process if we haven't seen this frame yet
                    if current_frame_id > last_frame_id:
                        print(f"✅ [Consumer] Received Frame: {current_frame_id} | Data shape: {frame_view.shape}")
                        
                        # --- GUI Display Block ---
                        # If you are on a headless setup, this block might fail safely.
                        try:
                            display_frame = frame_view.copy() 
                            cv2.imshow("Zero-Copy Stream", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("[Consumer] 'q' pressed, exiting.")
                                break
                        except Exception:
                            pass # Fail silently if no GUI is available
                            
                        last_frame_id = current_frame_id
                else:
                    # Producer might be lagging, just wait a tick
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[Consumer] Caught Ctrl+C, stopping gracefully.")
        finally:
            cv2.destroyAllWindows()
            streamer.close()
            print("[Consumer] Exiting.")


    # ==========================================
    # Main Process Orchestration
    # ==========================================
    SHM_PATH = "/dev/shm/test_streamer.dat"
    SOURCE = "/dev/video0"
    
    if not os.path.exists(SOURCE):
        print(f"⚠️ File {SOURCE} not found. Falling back to webcam (0).")
        SOURCE = 0

    shm_ready_event = mp.Event()

    p = mp.Process(target=producer, args=(SHM_PATH, SOURCE, shm_ready_event))
    c = mp.Process(target=consumer, args=(SHM_PATH, shm_ready_event))

    try:
        p.start()
        c.start()
        
        # Actively monitor the processes. 
        # If the video finishes (producer dies), this loop breaks and cleans up the consumer.
        while p.is_alive() and c.is_alive():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        print("[Main] Cleaning up processes...")
        # Terminate them safely if they are hung
        if p.is_alive():
            p.terminate()
            p.join()
        if c.is_alive():
            c.terminate()
            c.join()
        print("[Main] Cleanup complete. Ready for next run.")



def multi_stream():
    
    import multiprocessing as mp
    import cv2
    import os
    import time
    import numpy as np

    SHM_PATHS = [f"/dev/shm/test_streamer_{i}.dat" for i in range(3)]
    SOURCES = [
        "/dev/video0",
        ]
    
    def producer(shm_path, video_source, ready_event):
        print(f"[Producer-{shm_path}] Starting up...")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"[Producer-{shm_path}] Error: Could not open video source {video_source}")
            ready_event.set() # Unblock consumer so it can exit safely
            return

        ret, frame = cap.read()
        if not ret:
            print(f"[Producer-{shm_path}] Error: Could not read first frame.")
            ready_event.set()
            return
            
        actual_shape = frame.shape
        print(f"[Producer-{shm_path}] Video shape detected as: {actual_shape}")

        streamer = SharedMemoryStreamer(
            file_path=shm_path, 
            shape=actual_shape, 
            dtype=np.uint8, 
            create_if_missing=True
        )
        
        ready_event.set()

        frame_id = 0
        try:
            while True:
                streamer.write(frame, metadata={"frame_id": frame_id})
                frame_id += 1
                time.sleep(0.033)
                
                ret, frame = cap.read()
                if not ret:
                    print(f"[Producer-{shm_path}] Video finished.")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n[Producer-{shm_path}] Caught Ctrl+C, stopping gracefully.")
        finally:
            cap.release()
            streamer.close()
            print(f"[Producer-{shm_path}] Exiting.")
    
    def consumer(shm_path, ready_event):
        print(f"[Consumer-{shm_path}] Waiting for producer to initialize memory...")
        ready_event.wait()
        
        time.sleep(0.1) 
        
        print(f"[Consumer-{shm_path}] Memory ready, connecting...")

        streamer = SharedMemoryStreamer(
            file_path=shm_path, 
            create_if_missing=False
        )

        if streamer.mmap_obj is None:
            print(f"[Consumer-{shm_path}] Could not connect to shared memory. Exiting.")
            return

        last_frame_id = -1

        try:
            while True:
                frame_view, meta = streamer.read(timeout=1.0)
                
                if frame_view is not None:
                    current_frame_id = meta.get('shm_frame_id', -1)
                    
                    if current_frame_id > last_frame_id:
                        print(f"✅ [Consumer-{shm_path}] Received Frame: {current_frame_id} | Data shape: {frame_view.shape}")
                        
                        try:
                            display_frame = frame_view.copy() 
                            cv2.imshow(f"Stream {shm_path}", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print(f"[Consumer-{shm_path}] 'q' pressed, exiting.")
                                break
                        except Exception:
                            pass 
                            
                        last_frame_id = current_frame_id
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n[Consumer-{shm_path}] Caught Ctrl+C, stopping gracefully.")
        finally:
            cv2.destroyAllWindows()
            streamer.close()
            print(f"[Consumer-{shm_path}] Exiting.")
    
        # ==========================================
    
    for source in SOURCES:
        if not os.path.exists(source):
            print(f"⚠️ File {source} not found. Falling back to webcam (0).")
            source = 0
    
    ready_events = [mp.Event() for _ in SHM_PATHS]
    producers = [mp.Process(target=producer, args=(shm_path, source, ready_event)) for shm_path, source, ready_event in zip(SHM_PATHS, SOURCES, ready_events)]
    consumers = [mp.Process(target=consumer, args=(shm_path, ready_event)) for shm_path, ready_event in zip(SHM_PATHS, ready_events)]

    try:
        for p in producers:
            p.start()
        for c in consumers:
            c.start()
        
        while any(p.is_alive() for p in producers) and any(c.is_alive() for c in consumers):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        print("[Main] Cleaning up processes...")
        for p in producers:
            if p.is_alive():
                p.terminate()
                p.join()
        for c in consumers:
            if c.is_alive():
                c.terminate()
                c.join()
        print("[Main] Cleanup complete. Ready for next run.")



if __name__ == "__main__":
    # single_stream()
    multi_stream()
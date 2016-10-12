using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Denoise
{
    public class Vec3
    {
        public readonly int X;
        public readonly int Y;
        public readonly int Z;

        public Vec3()
        {
        }

        public Vec3(Vec3 v)
        {
            X = v.X;
            Y = v.Y;
            Z = v.Z;
        }

        public Vec3(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static double operator *(Vec3 v1, Vec3 v2)
        {
            return v1.X*v2.X + v1.Y*v2.Y + v1.Z*v2.Z;
        }

        public static Vec3 operator -(Vec3 v1, Vec3 v2)
        {
            return new Vec3(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z);
        }

        private static int CombineHashCodes(int h1, int h2)
        {
            return (h1 << 5) + h1 ^ h2;
        }

        private static int CombineHashCodes(int h1, int h2, int h3)
        {
            return CombineHashCodes(CombineHashCodes(h1, h2), h3);
        }

        public override bool Equals(object obj)
        {
            var o = obj as Vec3;
            if (o == null)
                return false;
            return o.X == X && o.Y == Y && o.Z == Z;
        }

        public override int GetHashCode()
        {
            return CombineHashCodes(X, Y, Z);
        }

        public bool IsColor()
        {
            return X >= 0 && X <= 255 && Y >= 0 && Y <= 255 && Z >= 0 && Z <= 255;
        }
    }

    public class Program
    {
        public static bool InField(Vec3[][] src, int x, int y)
        {
            return x >= 0 && x < src.Length && y >= 0 && y < src[0].Length;
        }

        public static IEnumerable<Vec3> GetNeighbours(Vec3[][] src, int x, int y, bool eightNeighbours)
        {
            var coords = eightNeighbours
                ? new[] { Tuple.Create(-1, -1), Tuple.Create(-1, 0), Tuple.Create(-1, 1), Tuple.Create(0, -1), Tuple.Create(0, 1), Tuple.Create(1, -1), Tuple.Create(1, 0), Tuple.Create(1, 1)}
                : new[] { Tuple.Create(-1, 0), Tuple.Create(1, 0), Tuple.Create(0, -1), Tuple.Create(0, 1)};
            return coords
                .Where(c => InField(src, x + c.Item1, y + c.Item2))
                .Select(c => src[x + c.Item1][y + c.Item2]);
        }

        public static IEnumerable<Vec3> GetNeighbourColors(Vec3[][] src, int x, int y, bool eightNeighbours, int eps)
        {
            var res = new HashSet<Vec3>();
            var neighbours = new List<Vec3>(GetNeighbours(src, x, y, eightNeighbours)) {src[x][y]};
            foreach (var neighbour in neighbours)
                for (int i = -eps; i < eps + 1; i++)
                    for (int j = -eps; j < eps + 1; j++)
                        for (int k = -eps; k < eps + 1; k++)
                        {
                            var t = new Vec3(neighbour.X + i, neighbour.Y + j, neighbour.Z + k);
                            if (t.IsColor())
                                res.Add(t);
                        }
            return res;
        }

        public static Vec3[][] RestoreImage(Vec3[][] src, double covar = 100, double maxDiff = 600,
            double weightDiff = 0.02, int iterations = 20, bool eightNeighbours = false, int eps = 1)
        {
            var buffer = new Vec3[2][][];
            var backbuffer = new Vec3[src.Length][];
            var zeroes = new Vec3[src.Length][];
            for (int i = 0; i < src.Length; i++)
            {
                var l = new List<Vec3>();
                var z = new List<Vec3>();
                for (int j = 0; j < src[0].Length; j++)
                {
                    l.Add(new Vec3(src[i][j]));
                    z.Add(new Vec3(0, 0, 0));
                }
                backbuffer[i] = l.ToArray();
                zeroes[i] = z.ToArray();
            }
            buffer[0] = backbuffer;
            buffer[1] = zeroes;

            var it = 0;
            var maxIt = iterations*src.Length;

            var current = false;
            var vMax = src.Length*src[0].Length*(256*256/(2.0*covar) + 4*weightDiff*maxDiff);
            for (int i = 0; i < iterations; i++)
            {
                current = !current;
                var s = Convert.ToInt32(!current);
                var d = Convert.ToInt32(current);
                Parallel.For(0, src.Length, x =>
                {
                    it++;
                    Console.Write($"\r{it/(double)maxIt*100:F}%");
                    for (int y = 0; y < src[0].Length; y++)
                    {
                        var vLocal = vMax;
                        var minVal = new Vec3();
                        foreach (var val in GetNeighbourColors(src, x, y, eightNeighbours, eps))
                        {
                            var v = val - src[x][y];
                            var vData = v * v / (2 * covar);
                            var vDiff = 0.0;

                            foreach (var e in GetNeighbours(buffer[s], x, y, eightNeighbours))
                            {
                                v = val - e;
                                vDiff += Math.Min(v * v, maxDiff);
                            }
                            var vCurrent = vData + weightDiff * vDiff;

                            if (vCurrent < vLocal)
                            {
                                minVal = val;
                                vLocal = vCurrent;
                            }
                        }
                        buffer[d][x][y] = minVal;
                    }
                });
            }
            Console.WriteLine();
            return buffer[Convert.ToInt32(current)];
        }

        public static Vec3[][] LoadImage(string imageFile)
        {
            var bmp = new Bitmap(imageFile);
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            var ptr = bmpData.Scan0;
            var bytes = Math.Abs(bmpData.Stride) * bmp.Height;
            var internalBuffer = new byte[bytes];

            Marshal.Copy(ptr, internalBuffer, 0, bytes);
            var l = new List<Vec3[]>();
            for (int u = 0; u < bmp.Width; u++)
            {
                var lv = new List<Vec3>();
                for (int v = 0; v < bmp.Height; v++)
                {
                    var pos = (u + v*bmp.Width)*3;
                    var b = internalBuffer[pos];
                    var g = internalBuffer[pos + 1];
                    var r = internalBuffer[pos + 2];
                    lv.Add(new Vec3(r, g, b));
                }
                l.Add(lv.ToArray());
            }
            return l.ToArray();
        }

        public static void SaveImage(Vec3[][] src, string imageFile)
        {
            var frame = new byte[src.Length*src[0].Length*3];

            for (int u = 0; u < src.Length; u++)
                for (int v = 0; v < src[0].Length; v++)
                {
                    var pos = (u + v*src.Length)*3;
                    frame[pos] = (byte) src[u][v].Z;
                    frame[pos + 1] = (byte) src[u][v].Y;
                    frame[pos + 2] = (byte) src[u][v].X;
                }

            var pinnedArray = GCHandle.Alloc(frame, GCHandleType.Pinned);
            var pointer = pinnedArray.AddrOfPinnedObject();

            var bmp = new Bitmap(src.Length, src[0].Length, 3*src.Length, PixelFormat.Format24bppRgb, pointer);
            bmp.Save(imageFile, ImageFormat.Bmp);
        }

        public static void PrintStats(Vec3[][] noisy, int eps, bool eightNeighbours)
        {
            Console.WriteLine($"Denoise with eps = {eps}, eightNeighbours = {eightNeighbours}:");
            var sw = new Stopwatch();
            sw.Start();
            var restored = RestoreImage(noisy, eps: eps, eightNeighbours: eightNeighbours);
            sw.Stop();
            Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds/1000.0} seconds");
            Console.WriteLine();
            SaveImage(restored, $"denoised-eps{eps}-{(eightNeighbours ? 8 : 4)}neigh.jpg");
        }

        public static void Main(string[] args)
        {
            var noisy = LoadImage("taj-rgb-noise.jpg");
            PrintStats(noisy, 0, false);
            PrintStats(noisy, 0, true);
            PrintStats(noisy, 1, false);
            PrintStats(noisy, 1, true);
            PrintStats(noisy, 2, false);
            PrintStats(noisy, 2, true);
        }
    }
}

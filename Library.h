/*
Some Implementations, that I learned and applied at CSES website.

CSES Profile : https://cses.fi/user/72463
Command Used: cd "d:\CP-DSA\" && g++ -Ddasari -Wshadow -pedantic -Wall
"-Wl,--stack=268435456" -Wextra -O2 -std=c++14 0.cpp -o 0 && "d:\CP-DSA\"0
*/

#include <bits/stdc++.h>
using namespace std;

// Left right pointer Implementation of Segment Tree.
template <typename T>
class SegTree {
    T val;
    int tl, tr;
    SegTree *left, *right;

   public:
    SegTree(vector<T>& arr, int l, int r) : tl(l), tr(r) {
        if (l == r) {
            left = NULL, right = NULL;
            val = arr[l];
            return;
        }
        int m = (l + r) / 2;
        left = new SegTree(arr, l, m);
        right = new SegTree(arr, m + 1, r);
        pull();
    }

    void pull() { val = max(left->val, right->val); }

    T Q(int l, int r) {
        if (r < tl or l > tr) return -1e18;
        if (l <= tl and tr <= r) return val;
        return max(left->Q(l, r), right->Q(l, r));
    }

    void U(int pos, int upd) {
        if (tl == tr) {
            val = upd;
            return;
        }
        int tm = (tl + tr) / 2;
        pos <= tm ? left->U(pos, upd) : right->U(pos, upd);
        pull();
    }
};

// segmen Tree Lazy and set
// Just think about leaves and how to merge them.
template <typename T>
struct LazysegNode {
   private:
    T val = 0, set = -1, add = 0;
    int tl, tr;
    LazysegNode *left, *right;

   public:
    LazysegNode(int l, int r, vector<int>& ar) : tl(l), tr(r) {
        if (l == r) {
            val = ar[l];
            return;
        }
        int m = (l + r) / 2;
        left = new LazysegNode(l, m, ar);
        right = new LazysegNode(m + 1, r, ar);
        pull();
    }

    void upd(int l, int r, int sv, int av) {
        if (r < tl or tr < l) return;
        if (l <= tl and tr <= r) {
            compose(sv, av);
            return;
        }
        push();
        left->upd(l, r, sv, av), right->upd(l, r, sv, av);
        pull();
    }

    T qry(int l, int r) {
        if (r < tl or tr < l) return 0;
        if (l <= tl and tr <= r) return sum();
        push(), pull();
        return left->qry(l, r) + right->qry(l, r);
    }

   private:
    T sum() {
        return set == -1 ? val + add * (tr - tl + 1)
                         : (set + add) * (tr - tl + 1);
    }
    void pull() { val = left->sum() + right->sum(); }
    void compose(T sv, T av) { sv != -1 ? set = sv, add = av : add += av; }
    void push() {
        left->compose(set, add), right->compose(set, add);
        set = -1, add = 0;
    }
};

// Disjoint set Union
class DSU {
    int n;
    vector<int> par;

   public:
    DSU(int sz) {
        n = sz + 5;
        par.resize(n);
        for (int i = 0; i < n; i++) par[i] = i;
    }

    int findPar(int n) {
        if (par[n] == n) return n;
        return par[n] = findPar(par[n]);
    }

    bool merge(int a, int b) {
        a = findPar(a), b = findPar(b);
        if (a == b) return false;
        par[a] = b;
        return true;
    }
};

// log operation - medianStream
class MedianStream {
   private:
    multiset<int, greater<int>> sml;
    multiset<int> big;

   public:
    void balance() {
        while (sml.size() > big.size() + 1) {
            auto it = sml.begin();
            big.insert(*it);
            sml.erase(it);
        }
        while (sml.size() < big.size()) {
            auto it = big.begin();
            sml.insert(*it);
            big.erase(it);
        }
    }

    void insert(int val) {
        if (val <= *(sml.begin()))
            sml.insert(val);
        else
            big.insert(val);
        balance();
    }

    void remove(int val) {
        if (sml.count(val)) {
            sml.erase(sml.find(val));
        } else {
            big.erase(big.find(val));
        }
        balance();
    }

    int getMedian() {
        int ans = *(sml.begin());
        return ans;
    }
};

// O(n) preprocessing
class combinatorics {
   private:
    int mod;
    vector<int> fact, modInv, facInv;

   public:
    combinatorics(int n, int m) {
        n += 5;
        fact.resize(n), modInv.resize(n), facInv.resize(n);
        mod = m;
        fact[0] = facInv[0] = modInv[1] = 1;
        for (int i = 2; i < n; i++)
            modInv[i] = mod - 1LL * mod / i * modInv[mod % i] % mod;
        for (int i = 1; i < n; i++) {
            fact[i] = 1LL * i * fact[i - 1] % mod;
            facInv[i] = 1LL * modInv[i] * facInv[i - 1] % mod;
        }
    }
    int powr(int a, int n) {
        assert(n >= 0);
        int res = 1;
        for (; n; n /= 2) {
            if (n % 2) res = 1LL * res * a % mod;
            a = 1LL * a * a % mod;
        }
        return res;
    }
    int nCr(int n, int r) {
        if (n < r) return 0;
        return 1LL * fact[n] * facInv[n - r] % mod * facInv[r] % mod;
    }
    int fac(int n) { return fact[n]; }
    int modI(int n) { return modInv[n]; }
    int facI(int n) { return facInv[n]; }
};

class TRIE {
    struct TrieNode {
        int freq;
        TrieNode* child[2];
    };

    TrieNode* root;
    int hgt;

   public:
    TRIE(int h) {
        root = getNode();
        hgt = h;
    };

    TrieNode* getNode() {
        TrieNode* newNode = new TrieNode;
        newNode->freq = 0;
        newNode->child[0] = newNode->child[1] = NULL;
        return newNode;
    }

    void insert(int key) {
        TrieNode* temp = root;
        for (int i = hgt - 1; i >= 0; i--) {
            int bit = (key >> i) & 1;
            if (temp->child[bit] == NULL) temp->child[bit] = getNode();
            temp = temp->child[bit];
            temp->freq++;
        }
    }

    void search(int key) {
        TrieNode* temp = root;
        for (int i = hgt - 1; i >= 0; i--) {
            int bit = (key >> i) & 1;
            temp = temp->child[bit];
            if (temp == NULL) {
                cout << '\n';
                return;
            }
            cout << temp->freq;
        }
        cout << '\n';
    }

    void remove(int key) {
        TrieNode* temp = root;
        for (int i = hgt - 1; i >= 0; i--) {
            int bit = (key >> i) & 1;
            temp = temp->child[bit];
            temp->freq--;
        }
    }

    int Max_XOR(int val) {
        int ans = 0;
        TrieNode* temp = root;
        for (int i = hgt - 1; i >= 0; i--) {
            int bit = ((val >> i) & 1) ^ 1;

            if (temp->child[bit] == NULL or temp->child[bit]->freq == 0)
                bit ^= 1;
            else
                ans |= (1 << i);
            temp = temp->child[bit];
        }
        return ans;
    }
};

// Strongly Connected Components.
class SCC {
   private:
    int tot;
    vector<vector<int>> A, A_rev;
    vector<int> order, vis, root;

   public:
    SCC(vector<vector<int>> g, int n) {
        tot = n;
        A = g;
        A_rev.resize(n + 1);
        for (int i = 1; i <= n; i++) {
            for (int& x : A[i]) {
                A_rev[x].push_back(i);
            }
        }
        vis.resize(n + 1, 0);
        root.resize(n + 1);
        for (int i = 1; i <= n; i++) root[i] = i;
    }

    void find_scc(vector<vector<int>>& scc, int& cur) {
        scc.clear(), cur = 0;
        scc.resize(tot + 1);
        for (int i = 1; i <= tot; i++)
            if (!vis[i]) dfs1(i);
        for (int i = 1; i <= tot; i++) vis[i] = false;
        for (int i = tot - 1; i >= 0; i--) {
            if (!vis[order[i]]) {
                vector<int> comp;
                dfs2(order[i], comp);
                cur++;
                for (int& x : comp) root[x] = cur;
            }
        }
        for (int i = 1; i <= tot; i++)
            for (int& x : A[i]) {
                if (root[x] != root[i]) {
                    scc[root[i]].push_back(root[x]);
                }
            }
        scc.resize(cur + 1);
    }

   private:
    void dfs1(int n) {
        vis[n] = true;
        for (int& x : A[n]) {
            if (!vis[x]) dfs1(x);
        }
        order.push_back(n);
    }

    void dfs2(int n, vector<int>& comp) {
        comp.push_back(n);
        vis[n] = true;
        for (int& x : A_rev[n]) {
            if (!vis[x]) {
                dfs2(x, comp);
            }
        }
    }
};

class BIT {
    int n;
    vector<int> bit;

   public:
    BIT(int sz) {
        n = sz;
        bit.resize(n + 5, 0);
    }

    void update(int k, int diff) {
        for (; k <= n; k += k & -k) bit[k] += diff;
    }

    int get_pf(int k) {
        int sum = 0;
        for (; k; k -= k & -k) sum += bit[k];
        return sum;
    }

    int query(int L, int R) { return get_pf(R) - get_pf(L - 1); }
};

class FENWICK {
    int n;
    vector<int> bit, arr;

   public:
    void resize(int sz) {
        n = sz + 5;
        arr.resize(n, 0);
        bit.resize(n, 0);
    };

    void update(int k, int nval) {
        int diff = nval - arr[k];
        arr[k] += diff;
        for (; k < n; k += k & -k) bit[k] += diff;
    }

    int get_pf(int k) {
        int sum = 0;
        for (; k; k -= k & -k) sum += bit[k];
        return sum;
    }

    int query(int l, int r) { return get_pf(r) - get_pf(l - 1); }
};

class SegmentTreeRec {
    struct node {
        int val;
        node(int v = 0) : val(v){};
    };
    vector<node> seg;
    int n, init;

   public:
    SegmentTreeRec(vector<int>& arr, int initialize = 0) {
        n = arr.size();
        init = initialize;
        seg.resize(4 * n, node(init));
        build(1, 0, n - 1, arr);
    }
    void build(int v, int tl, int tr, vector<int>& arr) {
        if (tl == tr) {
            seg[v] = node(arr[tl]);
            return;
        }
        int tm = (tl + tr) / 2;
        build(2 * v, tl, tm, arr);
        build(2 * v + 1, tm + 1, tr, arr);
        seg[v] = merge(seg[2 * v], seg[2 * v + 1]);
    }
    void update(int pos, int val) { _U(pos, val, 1, 0, n - 1); }
    void _U(int pos, int val, int v, int tl, int tr) {
        if (tl == tr) {
            seg[v] = node(val);
            return;
        }
        int tm = (tl + tr) / 2;
        pos <= tm ? _U(pos, val, 2 * v, tl, tm)
                  : _U(pos, val, 2 * v + 1, tm + 1, tr);
        seg[v] = merge(seg[2 * v], seg[2 * v + 1]);
    }
    node _Q(int l, int r, int v, int tl, int tr) {
        if (l > r) return node(init);
        if (l == tl and r == tr) return seg[v];
        int tm = (tl + tr) / 2;
        return merge(_Q(l, min(r, tm), 2 * v, tl, tm),
                     _Q(max(l, tm + 1), r, 2 * v + 1, tm + 1, tr));
    }
    int query(int l, int r) { return _Q(l, r, 1, 0, n - 1).val; }
    node merge(node a, node b) { return node(a.val + b.val); }
};

class SparseTable {
    int n, hgt = 0;
    vector<vector<int>> table;

   public:
    SparseTable(vector<int>& arr) {
        n = arr.size();
        while ((1 << hgt) <= n) hgt++;
        table.resize(n, vector<int>(hgt, 0));
        for (int b = 0; b < hgt; b++) {
            for (int i = 0; i < n; i++) {
                table[i][b] = b == 0
                                  ? arr[i]
                                  : fun(table[i][b - 1],
                                        table[(i + (1 << (b - 1))) % n][b - 1]);
            }
        }
    }
    int query(int L, int R) {  // zero indexing;
        int dist = R - L + 1, ans = 0;
        for (int b = 0; b < hgt and dist > 0; b++) {
            if ((dist >> b) & 1) ans = fun(ans, table[L % n][b]), L += (1 << b);
        }
        return ans;
    }
    int fun(int a, int b) { return __gcd(a, b); }
};

// Coordinate Compressing.
class CompressNums {
   public:
    vector<int> compress(vector<int>& arr) {
        vector<int> nums = arr;
        sort(nums.begin(), nums.end());
        int sz = unique(nums.begin(), nums.end()) - nums.begin();
        for (int& x : arr)
            x = lower_bound(nums.begin(), nums.begin() + sz, x) - nums.begin();
        return arr;
    }
};

class SegmentTreeLazy {
    struct node {
        int val, lazy, assigned;
        node(int v = 0, int l = 0, int a = 0) : val(v), lazy(l), assigned(a){};
    };
    vector<node> seg;
    int n, init;

   public:
    SegmentTreeLazy(vector<int>& arr, int initialize = 0) {
        n = arr.size();
        init = initialize;
        seg.resize(4 * n, node(init));
        build(1, 0, n - 1, arr);
    }
    void build(int v, int tl, int tr, vector<int>& arr) {
        if (tl == tr) {
            seg[v] = node(arr[tl]);
            return;
        }
        int tm = (tl + tr) / 2;
        build(2 * v, tl, tm, arr);
        build(2 * v + 1, tm + 1, tr, arr);
        seg[v] = merge(seg[2 * v], seg[2 * v + 1]);
    }
    void propagate(int v, int tl, int tr) {
        int tm = (tl + tr) / 2;
        if (seg[v].assigned) {
            _U(tl, tm, seg[v].assigned, false, 2 * v, tl, tm);
            _U(tm + 1, tr, seg[v].assigned, false, 2 * v + 1, tm + 1, tr);
            seg[v].assigned = 0;
        } else if (seg[v].lazy) {
            _U(tl, tm, seg[v].lazy, true, 2 * v, tl, tm);
            _U(tm + 1, tr, seg[v].lazy, true, 2 * v + 1, tm + 1, tr);
            seg[v].lazy = 0;
        }
    }
    void update(int l, int r, int val, bool lzy) {
        _U(l, r, val, lzy, 1, 0, n - 1);
    }
    void _U(int l, int r, int val, bool lzy, int v, int tl, int tr) {
        if (l > r) return;
        if (l == tl and r == tr) {
            if (lzy) {
                seg[v].val += val * (tr - tl + 1);
                if (seg[v].assigned == 0)
                    seg[v].lazy += val;
                else
                    seg[v].assigned += val;
            } else
                seg[v].val = val * (tr - tl + 1), seg[v].lazy = 0,
                seg[v].assigned = val;
            return;
        }
        propagate(v, tl, tr);
        int tm = (tl + tr) / 2;
        _U(l, min(tm, r), val, lzy, 2 * v, tl, tm);
        _U(max(tm + 1, l), r, val, lzy, 2 * v + 1, tm + 1, tr);
        seg[v] = merge(seg[2 * v], seg[2 * v + 1]);
    }
    node _Q(int l, int r, int v, int tl, int tr) {
        if (l > r) return node(init);
        if (l == tl and r == tr) return seg[v];
        propagate(v, tl, tr);
        int tm = (tl + tr) / 2;
        return merge(_Q(l, min(r, tm), 2 * v, tl, tm),
                     _Q(max(l, tm + 1), r, 2 * v + 1, tm + 1, tr));
    }
    int query(int l, int r) { return _Q(l, r, 1, 0, n - 1).val; }
    node merge(node a, node b) { return node(a.val + b.val); };
};

class TreeUpLift {  // 1 - indexed
    vector<vector<int>> A, up;
    vector<int> hgt, sz;
    int lmt = 20, n, rt;

   public:
    TreeUpLift(int tot, int root) {
        n = tot++, rt = root;
        A.resize(tot), sz.resize(tot), hgt.resize(tot);
        up.resize(tot, vector<int>(20, 0));
    }
    void add_edge(int a, int b) { A[a].push_back(b), A[b].push_back(a); }
    void dfs(int n = 1, int p = 0, int h = 0) {
        sz[n] = 1, hgt[n] = h, up[n][0] = p;
        for (int& x : A[n])
            if (x != p) dfs(x, n, h + 1);
    }
    void uplift() {
        for (int b = 1; b < lmt; b++) {
            for (int i = 1; i <= n; i++) up[i][b] = up[up[i][b - 1]][b - 1];
        }
    }
    int kth_anc(int a, int k) {
        for (int i = 0; i < lmt; i++) {
            if ((k >> i) & 1) a = up[a][i];
        }
        return a;
    }
    int lca(int a, int b) {
        if (hgt[a] < hgt[b]) swap(a, b);
        a = kth_anc(a, hgt[a] - hgt[b]);
        if (a == b) return a;
        for (int i = lmt - 1; i >= 0; i--) {
            if (up[a][i] != up[b][i]) a = up[a][i], b = up[b][i];
        }
        return up[a][0];
    }
    int dist(int a, int b) { return hgt[a] + hgt[b] - 2 * hgt[lca(a, b)]; }
};

class SubTreeQry {  // 1 - indexed
    vector<vector<int>> A;
    vector<int> hgt, sz, in, val;
    int lmt = 20, n, rt, timer;
    FENWICK fn;

   public:
    SubTreeQry(int tot, vector<int>& v, int root = 1) {
        n = tot++, rt = root, timer = 1, val = v;
        A.resize(tot), sz.resize(tot), hgt.resize(tot), in.resize(tot);
        fn.resize(tot);
    }
    void add_edge(int a, int b) { A[a].push_back(b), A[b].push_back(a); }
    void dfs(int n = 1, int p = 0, int h = 0) {
        sz[n] = 1, hgt[n] = h;
        in[n] = timer++;
        fn.update(in[n], val[n]);
        for (int& x : A[n]) {
            if (x == p) continue;
            dfs(x, n, h + 1);
            sz[n] += sz[x];
        }
    }
    void upd(int k, int val) { fn.update(in[k], val); }
    int qry(int v) { return fn.query(in[v], in[v] + sz[v] - 1); }
};

// Iterative Implementation of SegmentTree.
class SegmentTreeIter {
    int n;
    vector<int> seg;

   public:
    void init(int sz) {
        n = sz;
        seg.resize(2 * n, 0);
    }
    int query(int L, int R, int ans = 0) {  // zero indexing;
        for (L += n, R += n; L <= R; L >>= 1, R >>= 1) {
            if (L & 1) ans = fun(ans, seg[L++]);
            if (R & 1 ^ 1) ans = fun(ans, seg[R--]);
        }
        return ans;
    }
    void update(int pos, int val) {
        seg[pos + n] = val;
        for (int i = pos + n; i > 0; i >>= 1)
            seg[i >> 1] = fun(seg[i], seg[i ^ 1]);
    }
    int fun(int a, int b) { return max(a, b); }
};

// Yay! Heavy Light Decomposition.
class HLD {  // 1 - indexed
    int n, rt, timer = 0;
    vector<vector<int>> A;
    vector<int> hgt, head, heavy, pos, par;
    SegmentTreeIter seg;

   public:
    HLD(int tot, int root = 1) {
        n = tot++, rt = root;
        A.resize(tot), par.resize(tot), hgt.resize(tot);
        head.resize(tot), heavy.resize(tot, -1), pos.resize(tot);
        seg.init(tot - 1);
    }
    void add_edge(int a, int b) { A[a].push_back(b), A[b].push_back(a); }
    int dfs(int n = 1, int p = 0, int h = 1) {
        hgt[n] = h, par[n] = p;
        int heavy_c, mx_csz = 0, tot_sz = 1;
        for (int& x : A[n]) {
            if (x != p) {
                int sz_x = dfs(x, n, h + 1);
                tot_sz += sz_x;
                if (sz_x > mx_csz) mx_csz = sz_x, heavy[n] = x;
            }
        }
        return tot_sz;
    }
    void decompose(int n = 1, int h = 1, int p = 0) {
        head[n] = h, pos[n] = timer++;
        if (heavy[n] != -1) decompose(heavy[n], h, n);
        for (int& x : A[n]) {
            if (x == p or x == heavy[n]) continue;
            decompose(x, x, n);
        }
    }
    void upd(int s, int x) { seg.update(pos[s], x); }
    int qry(int a, int b) {
        int ans = 0;
        while (head[a] != head[b]) {
            if (hgt[head[a]] < hgt[head[b]]) swap(a, b);
            ans = max(ans, seg.query(pos[head[a]], pos[a]));
            a = par[head[a]];
        }
        if (hgt[a] > hgt[b]) swap(a, b);
        return max(ans, seg.query(pos[a], pos[b]));
    }
};

// Matrix Computations.
class Matrix {
   public:
    void matrix_product(vector<vector<int>>& a, vector<vector<int>>& b,
                        int mod) {
        int n = a.size(), m = a[0].size(), l = b[0].size();
        assert(m == (int)b.size());
        vector<vector<int>> res(n, vector<int>(l, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                for (int k = 0; k < m; k++) {
                    res[i][j] += a[i][k] * b[k][j] % mod;
                    res[i][j] %= mod;
                }
            }
        }
        a = res;
    }
    void matpowr(vector<vector<int>>& mat, int n, int m) {
        int sz = mat.size();
        vector<vector<int>> unit(sz, vector<int>(sz, 0));
        for (int i = 0; i < sz; i++) unit[i][i] = 1;
        while (n) {
            if (n % 2) matrix_product(unit, mat, m);
            matrix_product(mat, mat, m);
            n /= 2;
        }
        mat = unit;
    }
};

class RollingHash {
    int n, P, M;
    vector<int> exp, hval;

   public:
    RollingHash(string& s, int p, int m) {
        P = p, M = m, n = s.size();
        exp.resize(n + 1, 1), hval.resize(n);
        for (int i = 1; i <= n; i++) exp[i] = exp[i - 1] * P % M;
        for (int i = 0; i < n; i++) {
            hval[i] = exp[i] * (s[i] - 'a' + 1) % M;
            if (i) hval[i] += hval[i - 1], hval[i] %= M;
        }
    }
    int hash_value(int l, int r) {
        int hash_val = M + hval[r];
        if (l) hash_val -= hval[l - 1];
        return hash_val * exp[n - r] % M;
    }

    int str_hash_val(string& s) {
        int hash_val = 0, len = s.size();
        for (int i = 0; i < len; i++)
            hash_val += exp[i] * (s[i] - 'a' + 1) % M, hash_val %= M;
        return hash_val * exp[n - len + 1] % M;
    }
};

vector<int> Zfunction(string s) {
    int n = s.size(), l = 0, r = 0;
    vector<int> z(n);
    for (int i = 1; i < n; i++) {
        z[i] = max(0, min(z[i - l], r - i + 1));
        while (i + z[i] < n and s[i + z[i]] == s[z[i]]) l = i, r = i + z[i]++;
    }
    return z;
}

string ManachersAlgo(string& txt) {
    string s = "?";
    for (char& x : txt) s += x, s += '?';
    int n = s.size(), l = 0, r = 0;
    vector<int> f(n);

    for (int i = 0; i < n; i++) {
        // l.....|..i..r
        int k = max(1, min(r - i + 1, f[l + r - i]));
        while (i - k >= 0 and i + k < n and s[i - k] == s[i + k]) k++;
        f[i] = --k;
        if (i + k > r) r = i + k, l = i - k;
    }

    int mxpos = max_element(f.begin(), f.end()) - f.begin();

    string mx_pal = "";

    for (int i = mxpos - f[mxpos]; i <= mxpos + f[mxpos]; i++)
        if (s[i] != '?') mx_pal += s[i];
    return mx_pal;
}

class MaxFlowMinCut {
    int n;
    vector<vector<int>> cap, adj;
    vector<int> par;

   public:
    MaxFlowMinCut(int sz) {
        n = sz++;
        par.resize(sz), adj.resize(sz);
        cap.resize(sz, vector<int>(sz));
    }

    void add_edge(int a, int b, int w) {
        cap[a][b] += w;
        // cap[b][a] += w;  // uncomment this if directed
        if (cap[a][b] != w) return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    int bfs(int src, int des) {
        for (int i = 1; i <= n; i++) par[i] = -1;
        par[src] = -2;
        queue<pair<int, int>> qu;
        qu.push({src, INT_MAX});

        while (qu.size()) {
            int cur = qu.front().first, flow = qu.front().second;
            qu.pop();
            for (int& nxt : adj[cur]) {
                if (par[nxt] == -1 and cap[cur][nxt]) {
                    par[nxt] = cur;
                    if (nxt == des) return min(flow, cap[cur][nxt]);
                    qu.push({nxt, min(flow, cap[cur][nxt])});
                }
            }
        }
        return 0;
    }

    void dfs(int n, vector<bool>& vis) {
        if (vis[n]) return;
        vis[n] = true;
        for (int& x : adj[n])
            if (cap[n][x]) dfs(x, vis);
    }

    int max_flow(int src, int des) {
        int flow = 0, new_flow = 0;
        while (new_flow = bfs(src, des)) {
            int cur = des, prev = par[des];
            flow += new_flow;
            while (cur != src) {
                cap[prev][cur] -= new_flow;
                cap[cur][prev] += new_flow;
                cur = prev;
                prev = par[prev];
            }
        }
        return flow;
    }

    vector<array<int, 3>> min_cut(int src, int des) {
        vector<bool> vis(n + 1, 0);
        vector<array<int, 3>> cut;
        int flow = max_flow(src, des);
        dfs(src, vis);
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (vis[i] and !vis[j] and cap[j][i])
                    cut.push_back({i, j, cap[j][i]});
            }
        }
        return cut;
    }
};

class Kstream {
    int k, sum = 0;
    multiset<int> hi;
    multiset<int, greater<int>> lo;

   public:
    Kstream(int sz) { k = sz; };

    void insert(int n) {
        if (hi.size() < k)
            sum += n, hi.insert(n);
        else {
            auto it = hi.begin();
            if (*it < n) {
                sum += n - *it;
                lo.insert(*it);
                hi.erase(it);
                hi.insert(n);
            } else {
                lo.insert(n);
            }
        }
    }

    void remove(int n) {
        if (lo.count(n))
            lo.erase(lo.find(n));
        else {
            hi.erase(hi.find(n));
            sum -= n;
            if (lo.size()) {
                auto it = lo.begin();
                sum += *it;
                hi.insert(*it);
                lo.erase(it);
            }
        }
    }

    int get() { return sum; }
};

// class DiagonalMatrix {
//     int n, m;
//     vector<vector<int>> mat, pf;

//    public:
//     DiagonalMatrix(vector<vector<int>> &grid) {
//         mat = grid, pf = grid;
//         n = mat.size(), m = mat[0].size();
//         forn1(i, n - 1) forn1(j, m - 1) pf[i][j] += pf[i - 1][j - 1];
//     }
//     int query(int ti, int tj, int bi, int bj) {
//         assert(tj - ti == bj - bi);
//         int sum = pf[bi][bj];
//         if (ti and tj) sum -= pf[ti - 1][tj - 1];
//         return sum;
//     }
//     int ask(int i, int j, int dist) {
//         int ti = i - dist, tj = j;
//         int bi = i, bj = j + dist;
//         if (ti < 0) tj -= ti, ti = 0;
//         if (bj >= m) bi -= (bj - m + 1), bj = m - 1;
//         if (tj == j) ti++, tj++;
//         if (tj > bj or ti > bi or mat[i][j] == 0) return 0;
//         return query(ti, tj, bi, bj);
//     }
//     void rotate_clockwise() {
//         vvi temp(m, vi(n));
//         forn(i, n) forn(j, m) temp[j][n - 1 - i] = mat[i][j];
//         swap(n, m);
//         pf = temp, mat = temp;
//         forn1(i, n - 1) forn1(j, m - 1) pf[i][j] += pf[i - 1][j - 1];
//     }
// };

namespace gg {
struct range {
    int L, R, id;
    static int counter;
    range(int l = 0, int r = 0) : L(l), R(r){};
    friend istream& operator>>(istream& c, range& r);
    friend ostream& operator<<(ostream& c, const range& r);
    friend bool operator<(const range& a, const range& b) {
        if (a.R < b.R) return true;
        if (a.R == b.R) return a.L > b.L;
        return false;
    };
};

int range::counter = 0;
istream& operator>>(istream& c, range& r) {
    c >> r.L >> r.R;
    r.id = range::counter++;
    return c;
}
ostream& operator<<(ostream& c, const range& r) {
    c << "[" << r.L << " - " << r.R << "] - " << r.id;
    return c;
}
}  // namespace gg

// int main() {
//     cout << "Hello World\n";
//     return 0;
// }